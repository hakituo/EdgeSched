"""V0 单个消融单元：一个独立子进程里跑一个单元并写出 JSON。

LLM 负载与 S0 同口径：同一 GGUF、C512（n_ctx=640）、同一 prompt/系统提示/模板渲染、
temperature=0、seed=42、最多 128 token、flash_attn、整个进程绑到 LLM 核上
（S0 ``process_affinity_scope`` 的做法）。每轮记录每个 token 的交付时刻。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.elastic_residency.colocation.v0_cells import (  # noqa: E402
    CELLS_BY_NAME,
    Cell,
    decode_rate,
    tts_rtf_during,
)
from experiments.elastic_residency.colocation.v0_tts_loop import (  # noqa: E402
    DEFAULT_REF_AUDIO,
    DEFAULT_TTS_MODEL,
    SynthLoop,
    load_synth,
    run_for,
)

S0_PROMPT = "请用三到五句话介绍边缘计算的优势。"
STEADY_WAIT_S = 3.0


def _cpus(text: str) -> list[int]:
    return [int(cpu) for cpu in text.split(",") if cpu.strip()]


def gpu_used_mb() -> float | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return round(pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024 / 1024, 1)
    except Exception:  # pragma: no cover - 无 NVML 时不阻断
        return None


def open_llm(args: argparse.Namespace, n_threads: int) -> Any:
    """复用 S0 的 token 源构造（模板渲染、停止词与身份校验一致）。"""

    from experiments.elastic_residency.s0_pipeline.config import DIRECT_ANSWER_SYSTEM_PROMPT
    from experiments.elastic_residency.s0_pipeline.token_source import LlamaTokenSource

    source = LlamaTokenSource(
        model_path=args.llm_model,
        n_ctx=640,
        context_tokens=512,
        generation_budget_tokens=128,
        max_output_tokens=128,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=n_threads,
        seed=42,
        prompt=S0_PROMPT,
        system_prompt=DIRECT_ANSWER_SYSTEM_PROMPT,
        temperature=0.0,
        model_sha256=args.model_sha256,
        bridge_maxsize=16,
        recorder=None,
    )
    source.open()
    return source


def generate_once(source: Any) -> dict[str, Any]:
    """跑一轮生成，逐 token 记交付时刻（每个流式 chunk 为一个 token）。"""

    start = time.monotonic_ns()
    token_ns: list[int] = []
    text_parts: list[str] = []
    stream = source._llm.create_completion(
        prompt=source.rendered_prompt,
        max_tokens=128,
        temperature=0.0,
        seed=42,
        stream=True,
        stop=source._stop or None,
    )
    for chunk in stream:
        token_ns.append(time.monotonic_ns())
        text_parts.append(chunk["choices"][0].get("text") or "")
    end = time.monotonic_ns()
    rate = decode_rate(token_ns)
    text = "".join(text_parts)
    return {
        "start_ns": start,
        "end_ns": end,
        "tokens": len(token_ns),
        "ttft_s": round((token_ns[0] - start) / 1e9, 4) if token_ns else None,
        "decode_tok_s": round(rate, 3) if rate else None,
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "token_ns": token_ns,
    }


def start_tts_process(args: argparse.Namespace, affinity: list[int]) -> tuple[Any, Path]:
    out = Path(args.out).with_suffix(".tts_proc.json")
    cmd = [
        sys.executable,
        "-m",
        "experiments.elastic_residency.colocation.v0_tts_loop",
        "--out",
        str(out),
        "--affinity",
        ",".join(str(cpu) for cpu in affinity),
        "--tts-model",
        args.tts_model,
        "--ref-audio",
        args.ref_audio,
    ]
    proc = subprocess.Popen(
        cmd, cwd=PROJECT_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if line.strip() == "READY":
            return proc, out
        if not line and proc.poll() is not None:
            break
    proc.kill()
    raise RuntimeError("TTS 子进程未就绪")


def run_cell(cell: Cell, args: argparse.Namespace) -> dict[str, Any]:
    llm_cpus, tts_cpus = _cpus(args.llm_affinity), _cpus(args.tts_affinity)
    os.sched_setaffinity(0, llm_cpus)
    result: dict[str, Any] = {"cell": cell.to_dict(), "affinity": sorted(os.sched_getaffinity(0))}
    if cell.torch_threads is not None:
        import torch

        torch.set_num_threads(cell.torch_threads)
    loop: SynthLoop | None = None
    proc = proc_out = None
    if cell.tts_mode in ("idle", "thread"):
        synth, meta = load_synth(args.tts_model, args.ref_audio)
        result["tts_load"] = {k: meta.get(k) for k in ("load_s", "graph_capture_s")}
        loop = SynthLoop(synth)
    if not cell.run_llm:
        run_for(loop, args.tts_alone_seconds)
        result["tts_alone_rtf"] = tts_rtf_during(loop.records[1:])
        result["tts_records"] = loop.records
        result["passed"] = loop.error is None and bool(loop.records)
        result["error"] = loop.error
        return result
    n_threads = cell.llm_threads or len(llm_cpus)
    result["llm_threads"] = n_threads
    source = open_llm(args, n_threads)
    if cell.tts_mode == "thread":
        loop.start()
    elif cell.tts_mode.startswith("proc"):
        cpus = llm_cpus if cell.tts_mode == "proc_overlap" else tts_cpus
        proc, proc_out = start_tts_process(args, cpus)
        result["tts_proc_affinity"] = cpus
    if cell.tts_mode != "none":
        time.sleep(STEADY_WAIT_S)
    result["gpu_used_mb_steady"] = gpu_used_mb()
    generate_once(source)  # 预热轮，丢弃
    reps = [generate_once(source) for _ in range(args.reps)]
    if loop is not None and cell.tts_mode == "thread":
        loop.stop()
        result["tts_error"] = loop.error
        records = loop.records
    elif proc is not None:
        proc.stdin.close()
        proc.wait(timeout=120)
        payload = json.loads(proc_out.read_text(encoding="utf-8"))
        records = payload["records"]
        result["tts_error"] = payload.get("error")
        result["tts_proc_load"] = payload.get("load_meta")
    else:
        records = []
    for rep in reps:
        rep["tts_rtf"] = tts_rtf_during(records, rep["start_ns"], rep["end_ns"]) if records else None
    try:
        import torch

        result["torch_threads_effective"] = torch.get_num_threads()
    except ImportError:  # pragma: no cover
        result["torch_threads_effective"] = None
    result["reps"] = reps
    result["passed"] = all(rep["decode_tok_s"] for rep in reps) and not result.get("tts_error")
    result["error"] = result.get("tts_error")
    source.close()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V0 单元子进程")
    parser.add_argument("--cell", required=True, choices=sorted(CELLS_BY_NAME))
    parser.add_argument("--out", required=True)
    parser.add_argument("--llm-affinity", required=True)
    parser.add_argument("--tts-affinity", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--llm-model", default=str(PROJECT_ROOT / "models/llm/Qwen3.5-9B-Q4_K_M.gguf"))
    parser.add_argument("--n-gpu-layers", type=int, default=24)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--tts-alone-seconds", type=float, default=30.0)
    parser.add_argument("--tts-model", default=DEFAULT_TTS_MODEL)
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    args = parser.parse_args(argv)
    try:
        result = run_cell(CELLS_BY_NAME[args.cell], args)
    except Exception as exc:  # 失败也落盘，计入分母
        result = {
            "cell": CELLS_BY_NAME[args.cell].to_dict(),
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
