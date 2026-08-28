#!/usr/bin/env python3
"""B1：Qwen3-TTS T0–T3 组件发现与单文本真实 smoke。"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.elastic_residency.contracts import (  # noqa: E402
    BackendEvidence,
    ExperimentArtifactWriter,
    ExperimentRunConfig,
    FailureRecord,
)
from experiments.tts.profiling_utils import (  # noqa: E402
    model_tree_hash,
    sha256_file,
)


CONFIG_NAMES = ("T0", "T1", "T2", "T3")


@dataclass(frozen=True)
class PlacementSpec:
    name: str
    talker_device: str
    speech_codec_device: str
    speaker_device: str
    unload_speaker_after_prompt: bool = False


PLACEMENTS = {
    "T0": PlacementSpec("T0", "cuda:0", "cuda:0", "cuda:0"),
    "T1": PlacementSpec("T1", "cuda:0", "cpu", "cuda:0"),
    "T2": PlacementSpec("T2", "cuda:0", "cpu", "cpu"),
    "T3": PlacementSpec("T3", "cuda:0", "cpu", "cuda:0", True),
}


def placement_for(name: str) -> PlacementSpec:
    try:
        return PLACEMENTS[name.upper()]
    except KeyError as exc:
        raise ValueError(f"未知 TTS 放置配置: {name}") from exc


def _module_inventory(module: Any) -> dict[str, Any]:
    if module is None:
        return {
            "present": False,
            "parameter_bytes": 0,
            "buffer_bytes": 0,
            "devices": [],
            "dtypes": [],
        }
    parameters = list(module.parameters(recurse=True))
    buffers = list(module.buffers(recurse=True))
    return {
        "present": True,
        "class": f"{type(module).__module__}.{type(module).__qualname__}",
        "parameter_count": sum(item.numel() for item in parameters),
        "parameter_bytes": sum(
            item.numel() * item.element_size() for item in parameters
        ),
        "buffer_bytes": sum(item.numel() * item.element_size() for item in buffers),
        "devices": sorted({str(item.device) for item in parameters + buffers}),
        "dtypes": sorted({str(item.dtype) for item in parameters + buffers}),
    }


def _component_inventory(wrapper: Any) -> dict[str, Any]:
    core = wrapper.model
    speech_tokenizer = core.speech_tokenizer
    return {
        "text_tokenizer": {
            "present": True,
            "class": (
                f"{type(wrapper.processor).__module__}."
                f"{type(wrapper.processor).__qualname__}"
            ),
            "devices": ["cpu"],
        },
        "talker": _module_inventory(core.talker),
        "speech_codec": _module_inventory(speech_tokenizer.model),
        "speaker_encoder": _module_inventory(core.speaker_encoder),
        "speech_codec_wrapper_device": str(speech_tokenizer.device),
    }


def _move_module(module: Any, device: str) -> None:
    import torch

    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    module.to(device=torch.device(device), dtype=dtype)


def _synchronize() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_process_used_mb(pid: int) -> float:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return 0.0
    total = 0.0
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            if int(parts[0]) == pid:
                total += float(parts[1])
        except ValueError:
            continue
    return total


def _wait_for_gpu_probe(timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            if completed.stdout.strip():
                return True
        except (OSError, subprocess.SubprocessError):
            pass
        time.sleep(0.1)
    return False


def _memory_snapshot(process: psutil.Process) -> dict[str, Any]:
    import torch

    snapshot = {
        "rss_bytes": process.memory_info().rss,
        "process_cpu_percent": process.cpu_percent(interval=None),
        "system_available_bytes": psutil.virtual_memory().available,
        "gpu_process_used_mb": _gpu_process_used_mb(os.getpid()),
        "torch_allocated_bytes": 0,
        "torch_reserved_bytes": 0,
        "torch_peak_allocated_bytes": 0,
        "torch_peak_reserved_bytes": 0,
    }
    if torch.cuda.is_available():
        snapshot.update(
            torch_allocated_bytes=torch.cuda.memory_allocated(),
            torch_reserved_bytes=torch.cuda.memory_reserved(),
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    return snapshot


def _build_xvector_prompt(wrapper: Any, ref_audio: Path) -> tuple[Any, float]:
    import librosa
    import numpy as np
    import torch
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    audio, sample_rate = librosa.load(ref_audio, sr=None, mono=True)
    if sample_rate != wrapper.model.speaker_encoder_sample_rate:
        audio = librosa.resample(
            y=audio.astype(np.float32),
            orig_sr=int(sample_rate),
            target_sr=wrapper.model.speaker_encoder_sample_rate,
        )
    speaker = wrapper.model.speaker_encoder
    if speaker is None:
        raise RuntimeError("speaker encoder 已卸载，不能重新构建 prompt")
    parameter = next(speaker.parameters())
    started = time.perf_counter()
    mels = mel_spectrogram(
        torch.from_numpy(audio).unsqueeze(0),
        n_fft=1024,
        num_mels=128,
        sampling_rate=24000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    ).transpose(1, 2)
    with torch.inference_mode():
        embedding = speaker(mels.to(parameter.device, dtype=parameter.dtype))[0]
    _synchronize()
    elapsed = time.perf_counter() - started
    return (
        VoiceClonePromptItem(
            ref_code=None,
            ref_spk_embedding=embedding,
            x_vector_only_mode=True,
            icl_mode=False,
            ref_text=None,
        ),
        elapsed,
    )


def _apply_placement(wrapper: Any, spec: PlacementSpec) -> None:
    import torch

    core = wrapper.model
    _move_module(core.talker, spec.talker_device)
    _move_module(core.speech_tokenizer.model, spec.speech_codec_device)
    core.speech_tokenizer.device = torch.device(spec.speech_codec_device)
    if core.speaker_encoder is not None:
        _move_module(core.speaker_encoder, spec.speaker_device)
    wrapper.device = torch.device(spec.talker_device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _actual_devices(inventory: dict[str, Any]) -> dict[str, str]:
    def only(component: str) -> str:
        entry = inventory[component]
        if not entry["present"]:
            return "unloaded"
        devices = entry["devices"]
        return devices[0] if len(devices) == 1 else ",".join(devices)

    return {
        "talker": only("talker"),
        "speech_codec": only("speech_codec"),
        "speaker_encoder": only("speaker_encoder"),
        "text_tokenizer": "cpu",
    }


def _placement_matches(spec: PlacementSpec, actual: dict[str, str]) -> bool:
    expected_speaker = (
        "unloaded" if spec.unload_speaker_after_prompt else spec.speaker_device
    )
    return (
        actual["talker"] == spec.talker_device
        and actual["speech_codec"] == spec.speech_codec_device
        and actual["speaker_encoder"] == expected_speaker
        and actual["text_tokenizer"] == "cpu"
    )


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    spec = placement_for(args.child_config)
    model_path = Path(args.model).expanduser().resolve()
    ref_audio = Path(args.ref_audio).expanduser().resolve()
    output_wav = Path(args.child_output_wav).expanduser().resolve()
    if not model_path.is_dir() or not ref_audio.is_file():
        raise FileNotFoundError("B1 child 找不到模型或参考音频")
    if not torch.cuda.is_available():
        raise RuntimeError("B1 真实 smoke 要求 CUDA")
    gpu_probe_ready = _wait_for_gpu_probe(args.gpu_probe_timeout)
    if not gpu_probe_ready:
        raise RuntimeError("B1 计时前未获得可用 NVIDIA GPU 样本")
    process = psutil.Process()
    process.cpu_percent(interval=None)  # 明确丢弃初始化 CPU 样本。
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    wrapper = Qwen3TTSModel.from_pretrained(
        str(model_path),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    _synchronize()
    load_s = time.perf_counter() - load_started
    inventory_loaded = _component_inventory(wrapper)
    memory_loaded = _memory_snapshot(process)

    placement_started = time.perf_counter()
    _apply_placement(wrapper, spec)
    placement_s = time.perf_counter() - placement_started
    inventory_placed = _component_inventory(wrapper)
    memory_placed = _memory_snapshot(process)

    prompt, prompt_build_s = _build_xvector_prompt(wrapper, ref_audio)
    prompt_device_before_generate = str(prompt.ref_spk_embedding.device)
    if spec.unload_speaker_after_prompt:
        wrapper.model.speaker_encoder = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    inventory_ready = _component_inventory(wrapper)
    memory_ready = _memory_snapshot(process)
    actual_devices = _actual_devices(inventory_ready)

    tokenize_started = time.perf_counter()
    input_text = wrapper._build_assistant_text(args.text)
    input_ids = wrapper._tokenize_texts([input_text])
    tokenize_s = time.perf_counter() - tokenize_started
    prompt_dict = wrapper._prompt_items_to_voice_clone_prompt([prompt])
    generation_kwargs = wrapper._merge_generate_kwargs(
        do_sample=False,
        subtalker_dosample=False,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    _synchronize()
    torch.cuda.reset_peak_memory_stats()
    talker_started = time.perf_counter()
    codes, _ = wrapper.model.generate(
        input_ids=input_ids,
        ref_ids=None,
        voice_clone_prompt=prompt_dict,
        languages=[args.language],
        non_streaming_mode=True,
        **generation_kwargs,
    )
    _synchronize()
    talker_generate_s = time.perf_counter() - talker_started
    memory_talker_peak = _memory_snapshot(process)

    codec_device = torch.device(spec.speech_codec_device)
    _synchronize()
    torch.cuda.reset_peak_memory_stats()
    transfer_started = time.perf_counter()
    decode_codes = [code.to(codec_device) for code in codes]
    _synchronize()
    code_transfer_s = time.perf_counter() - transfer_started
    memory_transfer_peak = _memory_snapshot(process)
    torch.cuda.reset_peak_memory_stats()
    codec_started = time.perf_counter()
    wavs, sample_rate = wrapper.model.speech_tokenizer.decode(
        [{"audio_codes": code} for code in decode_codes]
    )
    _synchronize()
    codec_decode_s = time.perf_counter() - codec_started
    memory_codec_peak = _memory_snapshot(process)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, wavs[0], sample_rate, subtype="PCM_16")
    synthesis_s = tokenize_s + talker_generate_s + code_transfer_s + codec_decode_s
    audio_duration_s = len(wavs[0]) / sample_rate
    memory_final = _memory_snapshot(process)
    return {
        "config": spec.name,
        "passed": (
            _placement_matches(spec, actual_devices)
            and output_wav.is_file()
            and audio_duration_s > 0
        ),
        "backend_kind": "real_qwen3_tts_component_split",
        "fallback_occurred": False,
        "gpu_probe_ready_before_timing": gpu_probe_ready,
        "cpu_initial_sample_discarded": True,
        "text": args.text,
        "text_chars": len(args.text),
        "placement": asdict(spec),
        "actual_devices": actual_devices,
        "inventory_loaded": inventory_loaded,
        "inventory_placed": inventory_placed,
        "inventory_ready": inventory_ready,
        "prompt_device_before_generate": prompt_device_before_generate,
        "load_s": load_s,
        "placement_s": placement_s,
        "prompt_build_s": prompt_build_s,
        "text_tokenize_s": tokenize_s,
        "talker_generate_s": talker_generate_s,
        "code_transfer_s": code_transfer_s,
        "codec_decode_s": codec_decode_s,
        "synthesis_s": synthesis_s,
        "ttfa_s": synthesis_s,
        "audio_duration_s": audio_duration_s,
        "rtf": synthesis_s / audio_duration_s,
        "sample_rate": sample_rate,
        "codec_code_shapes": [list(code.shape) for code in codes],
        "output_wav": str(output_wav),
        "memory_loaded": memory_loaded,
        "memory_placed": memory_placed,
        "memory_ready": memory_ready,
        "memory_talker_peak": memory_talker_peak,
        "memory_transfer_peak": memory_transfer_peak,
        "memory_codec_peak": memory_codec_peak,
        "memory_final": memory_final,
    }


def _git_source(
    args: argparse.Namespace,
    artifact_root: Path,
) -> tuple[dict[str, Any], bool]:
    dirty = (
        args.source_git_dirty == "true"
        if args.source_git_dirty is not None
        else None
    )
    evidence = {
        "commit": args.source_git_commit or None,
        "dirty": dirty,
        "status_entry_count": args.source_git_status_entry_count,
        "snapshot": None,
        "snapshot_sha256": None,
    }
    snapshot_complete = False
    if args.source_git_snapshot:
        source = Path(args.source_git_snapshot).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"B1 源码快照不存在: {source}")
        digest = sha256_file(source)
        if (
            args.source_git_snapshot_sha256
            and digest != args.source_git_snapshot_sha256.lower()
        ):
            raise ValueError("B1 源码快照 SHA-256 与声明不一致")
        suffix = source.suffix if source.suffix else ".bin"
        destination = artifact_root / f"source_git_snapshot{suffix}"
        if source != destination.resolve():
            shutil.copyfile(source, destination)
        evidence["snapshot"] = destination.name
        evidence["snapshot_sha256"] = digest
        snapshot_complete = True
    complete = bool(args.source_git_commit) and (
        (dirty is False and args.source_git_status_entry_count == 0)
        or (
            dirty is True
            and bool(args.source_git_status_entry_count)
            and snapshot_complete
        )
    )
    return evidence, complete


def run_parent(args: argparse.Namespace) -> Path:
    model_path = Path(args.model).expanduser().resolve()
    ref_audio = Path(args.ref_audio).expanduser().resolve()
    model_sha256, model_files = model_tree_hash(model_path)
    ref_sha256 = sha256_file(ref_audio)
    writer = ExperimentArtifactWriter(args.output_root, args.run_id)
    source_git, git_provenance_complete = _git_source(
        args, writer.layout.root
    )
    config_names = tuple(name.upper() for name in args.configs)
    config = ExperimentRunConfig(
        experiment_name="tts_component_offload_b1",
        run_id=args.run_id,
        backends=(
            BackendEvidence(
                component="tts",
                backend_kind="real_qwen3_tts_component_split",
                is_real=True,
                model_path=str(model_path),
                model_sha256=model_sha256,
            ),
        ),
        parameters={
            "configs": config_names,
            "text": args.text,
            "language": args.language,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "x_vector_only_mode": True,
            "gpu_probe_timeout_s": args.gpu_probe_timeout,
            "cpu_initial_sample_discarded": True,
            "cuda_peak_reset_per_measured_phase": True,
        },
    )
    writer.initialize(
        config,
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git": {"source_copy": source_git},
            "git_provenance_complete": git_provenance_complete,
            "model_file_count": model_files,
            "reference_audio": str(ref_audio),
            "reference_audio_sha256": ref_sha256,
        },
    )
    records: list[dict[str, Any]] = []
    for config_name in config_names:
        placement_for(config_name)
        child_json = writer.layout.root / f"{config_name}_child.json"
        child_wav = writer.layout.root / "audio" / f"{config_name}.wav"
        child_log = writer.layout.root / f"{config_name}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-config",
            config_name,
            "--model",
            str(model_path),
            "--ref-audio",
            str(ref_audio),
            "--text",
            args.text,
            "--language",
            args.language,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--repetition-penalty",
            str(args.repetition_penalty),
            "--seed",
            str(args.seed),
            "--gpu-probe-timeout",
            str(args.gpu_probe_timeout),
            "--child-output-json",
            str(child_json),
            "--child-output-wav",
            str(child_wav),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.child_timeout,
        )
        child_log.write_text(
            completed.stdout + "\n[stderr]\n" + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0 or not child_json.is_file():
            writer.append_failure(
                FailureRecord(
                    stage=config_name,
                    error_type="ChildProcessFailure",
                    message=f"returncode={completed.returncode}",
                    monotonic_ns=time.monotonic_ns(),
                    fatal=False,
                    attributes={"log": str(child_log)},
                )
            )
            records.append(
                {
                    "config": config_name,
                    "passed": False,
                    "error": f"child returncode={completed.returncode}",
                }
            )
            continue
        record = json.loads(child_json.read_text(encoding="utf-8"))
        records.append(record)
        writer.append_iteration(record)
    passed = (
        git_provenance_complete
        and len(records) == len(config_names)
        and all(record.get("passed", False) for record in records)
    )
    summary = {
        "passed": passed,
        "config_count": len(config_names),
        "passed_count": sum(bool(record.get("passed")) for record in records),
        "git_provenance_complete": git_provenance_complete,
        "records": records,
    }
    writer.write_summary(summary)
    csv_path = writer.layout.summary_csv
    fields = [
        "config",
        "passed",
        "talker_device",
        "speech_codec_device",
        "speaker_encoder_device",
        "load_s",
        "placement_s",
        "prompt_build_s",
        "text_tokenize_s",
        "talker_generate_s",
        "code_transfer_s",
        "codec_decode_s",
        "ttfa_s",
        "audio_duration_s",
        "rtf",
        "gpu_process_used_mb_ready",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        csv_writer = csv.DictWriter(file_obj, fieldnames=fields)
        csv_writer.writeheader()
        for record in records:
            actual = record.get("actual_devices", {})
            csv_writer.writerow(
                {
                    "config": record.get("config"),
                    "passed": record.get("passed"),
                    "talker_device": actual.get("talker"),
                    "speech_codec_device": actual.get("speech_codec"),
                    "speaker_encoder_device": actual.get("speaker_encoder"),
                    "load_s": record.get("load_s"),
                    "placement_s": record.get("placement_s"),
                    "prompt_build_s": record.get("prompt_build_s"),
                    "text_tokenize_s": record.get("text_tokenize_s"),
                    "talker_generate_s": record.get("talker_generate_s"),
                    "code_transfer_s": record.get("code_transfer_s"),
                    "codec_decode_s": record.get("codec_decode_s"),
                    "ttfa_s": record.get("ttfa_s"),
                    "audio_duration_s": record.get("audio_duration_s"),
                    "rtf": record.get("rtf"),
                    "gpu_process_used_mb_ready": record.get(
                        "memory_ready", {}
                    ).get("gpu_process_used_mb"),
                }
            )
    return writer.layout.root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="/home/leslie/mvp_core/models/Qwen3-TTS-12Hz-0.6B-Base",
    )
    parser.add_argument(
        "--ref-audio",
        default="/home/leslie/mvp_core/models/tts_refs/ref_calm.wav",
    )
    parser.add_argument("--text", default="你好，这是组件放置验证。")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--configs", nargs="+", default=list(CONFIG_NAMES))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--child-timeout", type=float, default=900.0)
    parser.add_argument("--gpu-probe-timeout", type=float, default=5.0)
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "experiment_results" / "elastic_residency" / "b1"),
    )
    parser.add_argument(
        "--run-id",
        default=f"b1_tts_components_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--source-git-commit", default="")
    parser.add_argument("--source-git-dirty", choices=("true", "false"))
    parser.add_argument("--source-git-status-entry-count", type=int)
    parser.add_argument("--source-git-snapshot")
    parser.add_argument("--source-git-snapshot-sha256")
    parser.add_argument("--child-config", choices=CONFIG_NAMES)
    parser.add_argument("--child-output-json")
    parser.add_argument("--child-output-wav")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.child_config:
        try:
            record = run_child(args)
        except Exception as exc:
            if args.child_output_json:
                Path(args.child_output_json).write_text(
                    json.dumps(
                        {
                            "config": args.child_config,
                            "passed": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            raise
        Path(args.child_output_json).write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    output = run_parent(args)
    print(output)


if __name__ == "__main__":
    main()
