"""离线筛选 small 9B 的阶段组合；不启动 WSL、模型或 GPU。"""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics

from experiments.elastic_residency.analyze_results import (
    compute_g_safe, compute_h, extract_forge_profile, extract_tts_profiles,
)
from experiments.elastic_residency.analysis_input_merge import merge_llm_curve_summaries
from experiments.elastic_residency.compare_9b_support import SMALL_SHA


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiment_results/elastic_residency"
DEFAULTS = {
    "c2": "c2/c2_9b_small_20260907_143917",
    "c3": "c3/c3_9b_small_20260907_143917",
    "vision": "vision_endpoint/v1_vision_9b_small_profile_20260907_171635",
    "tts": "b2/b2_tts_pareto_matrix_20260831_direct_nvml_v2",
    "forge": "forge/forge_b1_reprofile_20260906_direct_nvml_v1",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_summary(summary):
    """Forge 把来源门放在 acceptance_checks，其他画像使用顶层字段。"""
    checks = summary.get("acceptance_checks", {})
    provenance = summary.get("git_provenance_complete", checks.get("git_provenance_complete"))
    if summary.get("passed") is not True or provenance is not True:
        raise ValueError("输入未通过或缺少来源")
    if checks and not all(value is True for value in checks.values()):
        raise ValueError("输入存在失败门禁")


def load_inputs(paths):
    """固定模型身份并核验源码/原始 child；历史辅助画像只作为筛选先验。"""
    sources, hashes, timelines = {}, {}, {}
    for role, root in paths.items():
        summary = read_json(root / "summary.json")
        validate_summary(summary)
        sources[role] = summary
        summary["run_id"] = root.name
        hashes[role] = {"path": str(root), "summary_sha256": hashlib.sha256(
            (root / "summary.json").read_bytes()).hexdigest()}
        if role not in ("c2", "c3", "vision"):
            continue
        source = read_json(root / "manifest.json")["git"]["source_copy"]
        digest = hashlib.sha256((root / source["snapshot"]).read_bytes()).hexdigest()
        if digest != source["snapshot_sha256"]:
            raise ValueError(f"{role} 源码快照不符")
        hashes[role]["source_snapshot_sha256"] = digest
        if role == "vision":
            gate = summary["endpoint_gate"]
            model_sha = gate["artifacts"]["main_model"]["sha256"]
            if not gate["passed"] or gate["artifacts"]["projector"]["sha256"] != (
                    "e36da9b9891e13a172b25028ad18b4fc63bc4d681cf391a44b2a9a3b51bc797a"):
                raise ValueError("视觉配对身份不符")
        else:
            backend = read_json(root / "run_config.json")["backends"][0]
            model_sha = backend["model_sha256"]
            if not backend["is_real"] or backend["fallback_occurred"]:
                raise ValueError("不是无回退的真实后端")
        if model_sha != SMALL_SHA:
            raise ValueError(f"{role} 不是选定 small 9B")
        records = summary["records"]
        if ([json.loads(line) for line in (root / "iterations.jsonl").read_text(encoding="utf-8").splitlines()]
                != records or (root / "failures.jsonl").read_text(encoding="utf-8").strip()):
            raise ValueError(f"{role} 原始记录不一致")
        if len(records) != summary["expected_records"]:
            raise ValueError(f"{role} 记录不完整")
        timelines[role] = []
        for record in records:
            prefix = (f"{record['config']}_{record['residency_state']}" if role == "vision"
                      else f"C{record['context_tokens']}_{record['config']}")
            child = read_json(root / f"{prefix}_r{record['run_index']}_child.json")
            if any(child.get(k) != v for k, v in record.items() if k != "matrix_execution_ordinal"):
                raise ValueError("child 与汇总不符")
            if not record["passed"] or not all(record["acceptance_checks"].values()):
                raise ValueError("原始门禁未通过")
            timeline = child["timeline"]
            if not timeline or any(p["system_swap_used_bytes"] != 0 for p in timeline):
                raise ValueError("缺少时间线或存在 swap")
            timelines[role].append(timeline)
    if hashes["c2"]["source_snapshot_sha256"] != hashes["c3"]["source_snapshot_sha256"]:
        raise ValueError("C2/C3 源码快照不同")
    merge_llm_curve_summaries(sources["c2"], sources["c3"], expected_config_count=16)
    return sources, hashes, timelines


def calculate(curve, tts, forge, vision_delta, idle_high, gpu_total=8151.0):
    """已测增量的保守 GPU 筛选；不推导主机内存可行或冻结 H。"""
    margin = compute_g_safe(curve, tts, idle_high)
    text = {}
    for context, configs in curve.items():
        text[context] = {name: compute_h(configs, profile["max_card_delta_mb"],
                                        margin["value_mb"], gpu_total, basis="max")
                         for name, profile in tts.items()}
    visual = {name: {"estimated_gpu_slack_mib": gpu_total - margin["value_mb"]
                     - vision_delta - profile["max_card_delta_mb"]}
              for name, profile in tts.items()}
    # 图像生成仅保留 GPU 数值诊断，16G 主机门未解除，不产生可执行组合。
    diffusion = {context: {name: compute_h(configs, forge + profile["max_card_delta_mb"],
                                           margin["value_mb"], gpu_total, basis="max")
                           for name, profile in tts.items()}
                 for context, configs in curve.items()}
    four = {name: gpu_total - margin["value_mb"] - vision_delta
            - profile["max_card_delta_mb"] - forge for name, profile in tts.items()}
    return {"provisional_g_safe": margin, "text_llm_tts_gpu_only": text,
            "full_c512_vision_tts_gpu_only": visual,
            "diffusion_tts_llm_gpu_diagnostic_only": diffusion,
            "full_c512_four_component_gpu_slack_mib": four,
            "diffusion_host_gate": "16G negative boundary retained; no automatic rerun",
            "scope": "GPU candidate estimates only; host memory, residuals and runtime compatibility pending"}


def run(args):
    paths = {role: Path(getattr(args, role)).resolve() for role in DEFAULTS}
    sources, hashes, timelines = load_inputs(paths)
    curve, baselines = {}, []
    for role in ("c2", "c3"):
        buckets = {}
        for r, timeline in zip(sources[role]["records"], timelines[role], strict=True):
            baseline = r["memory_baseline"]["gpu_card_used_mb"]
            baselines.append(baseline)
            key = (str(r["context_tokens"]), r["config"])
            buckets.setdefault(key, []).append(max(p["gpu_card_used_mb"] for p in timeline) - baseline)
        for (context, config), values in buckets.items():
            if len(values) != 5:
                raise ValueError("驻留每单元需五轮")
            curve.setdefault(context, {})[config] = {
                "mean": statistics.mean(values), "min": min(values), "max": max(values)}
    vision_deltas, costs = [], []
    for r, timeline in zip(sources["vision"]["records"], timelines["vision"], strict=True):
        if (r["config"], r["context_tokens"], r["n_ctx"], r["generation_tokens"]) != ("FULL", 512, 640, 128):
            raise ValueError("视觉筛选仅接受已测 FULL/C512/n_ctx640")
        baselines.append(r["memory_baseline"]["gpu_card_used_mb"])
        vision_deltas.append(max(p["gpu_card_used_mb"] for p in timeline) - baselines[-1])
        v = r["vision"]
        costs.append({"load_warm_s": v["session"]["projector_load_s"],
                      "encode_s": v["chunk_execution"]["projector_encode_total_s"],
                      "free_s": v["release"]["projector_free_s"]})
    if len(vision_deltas) != 5:
        raise ValueError("视觉需五轮画像")
    forge = extract_forge_profile(sources["forge"], None, None)
    result = calculate(curve, extract_tts_profiles(sources["tts"]),
                       forge["m_d_incremental_estimate_mb"], max(vision_deltas),
                       max(*baselines, forge["global_idle_high_water_mb"]))
    result.update(inputs=hashes, vision_costs=costs, llm_curve=curve,
                  unmeasured=["CPU TTS paired endpoints", "vision at mixed LLM tiers/longer contexts",
                              "host memory overlap", "persistent-session retention/early release"])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"已写入 {out}")
    print("仅离线 GPU 候选筛选；未运行模型，不冻结阈值，不重启 Diffusion 16G 停止矩阵。")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for role, path in DEFAULTS.items():
        parser.add_argument(f"--{role}", default=str(BASE / path))
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    parser.add_argument("--output", default=str(BASE / f"c3_analysis/small_stage_screen_{stamp}.json"))
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
