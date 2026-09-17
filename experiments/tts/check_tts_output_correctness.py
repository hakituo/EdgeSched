"""对 B2 画像产物做波形正确性核对：ASR 回转 + 文本比对。

在**主环境**运行（隔离环境不装 ASR，不为核对再改后端环境），读 B2 运行目录里的
``chunks.jsonl`` 与 ``audio/*.wav``，逐条回转后判定。ASR 口径由本模块自己钉住，
不复用辅助负载的 `WhisperAsr`：那个实例的固定参数改不得，而这里必须用简体偏置
的 ``initial_prompt``，否则 small-int8 模型会输出繁体，逐字比对会把简繁差异误判成
错误（2026-09-17 首次核对即踩到这个坑，10/15 条假失败）。

**判定分级**（``pass_rule`` 随结果一起写盘）：

- 硬失败：``empty``（转写为空）、``truncated``（明显短于原文）、``repeat``
  （出现异常重复 n-gram，波形塌陷或串句特征）、``mismatch``（匹配率 < 0.7）；
- ``partial``：0.7 ≤ 匹配率 < 0.9，属于小模型 int8 的谐音误识别（例如「尾延迟→伪延迟」），
  记账但不判失败，须人工抽查；
- ``ok``：匹配率 ≥ 0.9。

状态残留按**文本层**判定：同一句重复多次时转写应唯一。波形层另记
``waveform_bitwise_variants``——加速路径逐次波形本来就不位精确（``upstream_dynamic``
重复调用唯一，``cuda_graph_static`` 每轮不同），这是已知边界，不作为残留证据。

结果写运行目录的 ``correctness.json``，并把 ``summary.json`` 的
``correctness_check`` 从 ``pending_external_asr_pass`` 更新为实测结论。
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.tts.b2_profile_contract import (  # noqa: E402
    CORRECTNESS_REPEAT,
    TEXTS,
)

_KEEP = re.compile(r"[\u4e00-\u9fff0-9a-zA-Z]")
DEFAULT_WHISPER_MODEL = PROJECT_ROOT / "models" / "auxiliary" / "faster-whisper-small"
#: 简体偏置提示词；不加这句 small-int8 会输出繁体，导致大量假失败。
DEFAULT_INITIAL_PROMPT = "以下是普通话的句子，请用简体中文转写。"
PASS_RATIO = 0.9
PARTIAL_RATIO = 0.7
TRUNCATED_RATIO = 0.8
HARD_FAILURES = ("empty", "truncated", "repeat", "mismatch")


def normalize_text(text: str) -> str:
    """只保留中文与字母数字；标点与空格差异不算错误。"""

    return "".join(char.lower() for char in _KEEP.findall(text))


def matched_chars(expected: str, transcript: str) -> int:
    matcher = difflib.SequenceMatcher(a=expected, b=transcript)
    return sum(block.size for block in matcher.get_matching_blocks())


def repeated_ngrams(text: str, size: int = 4, limit: int = 3) -> list[str]:
    """出现次数超过 ``limit`` 的 n-gram。"""

    if len(text) < size:
        return []
    counts = Counter(text[index : index + size] for index in range(len(text) - size + 1))
    return sorted(gram for gram, count in counts.items() if count > limit)


def compare(expected: str, transcript: str) -> dict[str, Any]:
    """单条判定；纯函数，便于静态验证。"""

    expected_norm = normalize_text(expected)
    transcript_norm = normalize_text(transcript)
    matched = matched_chars(expected_norm, transcript_norm)
    ratio = matched / len(expected_norm) if expected_norm else 0.0
    repeats = repeated_ngrams(transcript_norm)
    if not transcript_norm:
        verdict = "empty"
    elif repeats:
        verdict = "repeat"
    elif ratio >= PASS_RATIO:
        verdict = "ok"
    elif ratio < PARTIAL_RATIO and len(transcript_norm) < TRUNCATED_RATIO * len(expected_norm):
        verdict = "truncated"
    elif ratio >= PARTIAL_RATIO:
        verdict = "partial"
    else:
        verdict = "mismatch"
    return {
        "verdict": verdict,
        "expected_chars": len(expected_norm),
        "transcript_chars": len(transcript_norm),
        "matched_chars": matched,
        "char_ratio": round(ratio, 4),
        "missing_chars": len(expected_norm) - matched,
        "extra_chars": len(transcript_norm) - matched,
        "repeat_ngrams": repeats,
        "transcript": transcript,
    }


def asr_library_versions() -> dict[str, str]:
    """ASR 仪器版本；不同环境的 ctranslate2 版本会改变匹配率，必须随产物记录。"""

    from importlib import metadata

    versions: dict[str, str] = {}
    for package in ("faster_whisper", "ctranslate2"):
        try:
            versions[package] = metadata.version(package)
        except Exception:  # noqa: BLE001 - 版本探测失败只记 unknown
            versions[package] = "unknown"
    return versions


class RoundtripAsr:
    """主环境 ASR 仪器；参数固定并随产物记录，重依赖惰性导入。"""

    def __init__(
        self,
        model_path: Path,
        *,
        cores: int,
        input_root: Path,
        language: str,
        initial_prompt: str,
    ) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            str(model_path),
            device="cpu",
            compute_type="int8",
            cpu_threads=cores,
            num_workers=1,
            local_files_only=True,
        )
        self.input_root = input_root
        self.language = language
        self.initial_prompt = initial_prompt

    def run(self, relative: str) -> dict[str, Any]:
        segments, info = self.model.transcribe(
            str(self.input_root / relative),
            language=self.language,
            beam_size=1,
            temperature=0,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt,
        )
        # 转写是惰性生成器，必须完全消费后才算完成。
        segments = list(segments)
        return {
            "text": "".join(segment.text for segment in segments).strip(),
            "audio_duration_s": info.duration,
            "segment_count": len(segments),
        }


def _load_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (run_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"{run_dir}/chunks.jsonl 没有测量记录")
    return rows


def check_run(
    run_dir: Path,
    *,
    whisper_model: Path,
    cores: int,
    language: str = "zh",
    initial_prompt: str = DEFAULT_INITIAL_PROMPT,
) -> dict[str, Any]:
    rows = _load_rows(run_dir)
    asr = RoundtripAsr(
        whisper_model,
        cores=cores,
        input_root=run_dir / "audio",
        language=language,
        initial_prompt=initial_prompt,
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        if "wav_path" not in row:
            raise ValueError("缺少 wav_path：正确性核对要求保存音频（去掉 --no-audio）")
        expected = row.get("expected_text") or TEXTS[row["text"]]
        relative = Path(row["wav_path"]).relative_to(run_dir / "audio").as_posix()
        outcome = asr.run(relative)
        records.append(
            {
                "path": row["path"],
                "text": row["text"],
                "call_index": row["call_index"],
                "pcm_sha256": row["pcm_sha256"],
                "audio_duration_s": row.get("audio_s"),
                "asr_duration_s": outcome.get("audio_duration_s"),
                "asr_segment_count": outcome.get("segment_count"),
                **compare(expected, outcome.get("text", "")),
            }
        )

    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        bucket = groups.setdefault(
            f"{record['path']}|{record['text']}",
            {"calls": 0, "verdicts": Counter(), "transcripts": set(), "pcm_sha256": set()},
        )
        bucket["calls"] += 1
        bucket["verdicts"][record["verdict"]] += 1
        bucket["transcripts"].add(normalize_text(record["transcript"]))
        bucket["pcm_sha256"].add(record["pcm_sha256"])
    summary_groups = {
        key: {
            "calls": value["calls"],
            "verdicts": dict(value["verdicts"]),
            "distinct_transcripts": len(value["transcripts"]),
            "distinct_pcm_sha256": len(value["pcm_sha256"]),
        }
        for key, value in groups.items()
    }

    residue = {
        key: {
            "calls": summary_groups[key]["calls"],
            "distinct_transcripts": summary_groups[key]["distinct_transcripts"],
            "waveform_bitwise_variants": summary_groups[key]["distinct_pcm_sha256"],
            "text_deterministic": summary_groups[key]["distinct_transcripts"] == 1,
            "waveform_note": (
                "加速路径逐次波形不位精确（上游路径重复调用唯一），"
                "因此判定只用文本层唯一性，波形变体数仅作记录"
            ),
        }
        for key in sorted(summary_groups)
        if key.endswith(f"|{CORRECTNESS_REPEAT}")
    }

    verdicts = Counter(record["verdict"] for record in records)
    hard = [record for record in records if record["verdict"] in HARD_FAILURES]
    partial = [record for record in records if record["verdict"] == "partial"]
    return {
        "checker": "faster_whisper_cpu_int8_roundtrip",
        "asr_library_versions": asr_library_versions(),
        "whisper_model": str(whisper_model),
        "whisper_model_present": whisper_model.exists(),
        "asr_language": language,
        "asr_initial_prompt": initial_prompt,
        "asr_initial_prompt_reason": (
            "简体偏置；不加时 small-int8 输出繁体，简繁差异会被逐字比对误判为错误"
        ),
        "pass_rule": (
            f"硬失败={'/'.join(HARD_FAILURES)}；partial（{PARTIAL_RATIO}<=ratio<{PASS_RATIO}）"
            "只记账须人工抽查；passed 要求无硬失败"
        ),
        "pass_ratio": PASS_RATIO,
        "partial_ratio": PARTIAL_RATIO,
        "call_count": len(records),
        "verdict_counts": dict(verdicts),
        "hard_failure_count": len(hard),
        "partial_count": len(partial),
        "failures": [
            {
                "path": record["path"],
                "text": record["text"],
                "call_index": record["call_index"],
                "verdict": record["verdict"],
                "char_ratio": record["char_ratio"],
                "missing_chars": record["missing_chars"],
                "extra_chars": record["extra_chars"],
                "repeat_ngrams": record["repeat_ngrams"],
                "transcript": record["transcript"],
            }
            for record in hard
        ],
        "partial_examples": [
            {
                "text": record["text"],
                "call_index": record["call_index"],
                "char_ratio": record["char_ratio"],
                "transcript": record["transcript"],
            }
            for record in partial
        ],
        "state_residue": residue,
        "all_ok": not hard,
        "groups": summary_groups,
        "records": records,
    }


def update_summary(run_dir: Path, report: dict[str, Any]) -> None:
    target = run_dir / "summary.json"
    if not target.is_file():
        return
    summary = json.loads(target.read_text(encoding="utf-8"))
    summary["correctness_check"] = (
        "completed_all_ok" if report["all_ok"] else "completed_with_failures"
    )
    summary["correctness_verdict_counts"] = report["verdict_counts"]
    summary["correctness_failure_count"] = report["hard_failure_count"]
    summary["correctness_partial_count"] = report["partial_count"]
    summary["correctness_checker"] = report["checker"]
    summary["correctness_pass_rule"] = report["pass_rule"]
    # 正确性核对是 passed 的最后一道门；来源完整性由画像自身记录，不在这里改
    summary["passed"] = bool(report["all_ok"])
    target.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--whisper-model", type=Path, default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--initial-prompt", default=DEFAULT_INITIAL_PROMPT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    report = check_run(
        run_dir,
        whisper_model=args.whisper_model,
        cores=args.cores,
        language=args.language,
        initial_prompt=args.initial_prompt,
    )
    (run_dir / "correctness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    update_summary(run_dir, report)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "all_ok": report["all_ok"],
                "verdict_counts": report["verdict_counts"],
                "partial_count": report["partial_count"],
                "state_residue": report["state_residue"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
