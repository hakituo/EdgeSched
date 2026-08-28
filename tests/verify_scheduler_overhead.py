import argparse
import asyncio
import json
import os
import platform
import sys
import time

import psutil


def _find_mvp_root() -> str:
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(12):
        if os.path.isdir(os.path.join(curr_dir, "services")) and os.path.isdir(os.path.join(curr_dir, "experiment_results")):
            return curr_dir

        candidate = os.path.join(curr_dir, "legacy", "mvp_core")
        if os.path.isdir(os.path.join(candidate, "services")) and os.path.isdir(os.path.join(candidate, "experiment_results")):
            return candidate

        parent = os.path.dirname(curr_dir)
        if parent == curr_dir:
            break
        curr_dir = parent

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rss_mb() -> float:
    proc = psutil.Process()
    return float(proc.memory_info().rss) / 1024.0 / 1024.0


def _default_model_path(mvp_root: str) -> str:
    env_path = str(os.getenv("XIAOYOU_LLM_MODEL_PATH", "") or "").strip()
    if env_path:
        return env_path
    candidate = os.path.join(mvp_root, "models", "qwen2.5-7b-instruct-q4_k_m.gguf")
    if os.path.exists(candidate):
        return candidate
    return ""


async def _stop_engine(engine) -> None:
    try:
        await engine.stop()
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="",
        help="结果输出路径（默认写到 experiment_results/supplementary/scheduler_overhead.json）",
    )
    parser.add_argument("--preload", action="store_true", help="额外测一次 preload_llm=True 的耗时/内存")
    parser.add_argument("--import_s_max", type=float, default=2.0)
    parser.add_argument("--start_s_max", type=float, default=0.2)
    parser.add_argument("--start_delta_mb_max", type=float, default=5.0)
    args = parser.parse_args()

    mvp_root = _find_mvp_root()
    if mvp_root not in sys.path:
        sys.path.insert(0, mvp_root)
    results_dir = os.path.join(mvp_root, "experiment_results", "supplementary")
    os.makedirs(results_dir, exist_ok=True)

    output_path = str(args.output or "").strip() or os.path.join(results_dir, "scheduler_overhead.json")

    rss_before_import = _rss_mb()
    t0 = time.perf_counter()
    from services.cpp_scheduler_engine import cpp_scheduler_engine
    t1 = time.perf_counter()
    rss_after_import = _rss_mb()

    if not getattr(cpp_scheduler_engine, "enabled", False):
        payload = {
            "name": "scheduler_overhead",
            "skipped": True,
            "reason": "cpp_scheduler_engine.enabled=False",
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "rss_before_import_mb": round(rss_before_import, 3),
            "rss_after_import_mb": round(rss_after_import, 3),
            "import_s": round(t1 - t0, 6),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    model_path = _default_model_path(mvp_root)
    start_cfg = {
        "model_path": model_path,
        "max_context_size": 1024,
        "n_gpu_layers": 0,
        "max_batch_size": 128,
        "verbose": False,
    }

    rss_before_start = _rss_mb()
    t2 = time.perf_counter()
    cpp_scheduler_engine.start(worker_count=1, gpu_config=start_cfg, preload_llm=False)
    t3 = time.perf_counter()
    rss_after_start = _rss_mb()
    asyncio.run(_stop_engine(cpp_scheduler_engine))

    payload = {
        "name": "scheduler_overhead",
        "skipped": False,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "thresholds": {
            "import_s_max": float(args.import_s_max),
            "start_s_max": float(args.start_s_max),
            "start_delta_mb_max": float(args.start_delta_mb_max),
        },
        "measurements": {
            "rss_before_import_mb": round(rss_before_import, 3),
            "rss_after_import_mb": round(rss_after_import, 3),
            "import_s": round(t1 - t0, 6),
            "rss_before_start_mb": round(rss_before_start, 3),
            "rss_after_start_mb": round(rss_after_start, 3),
            "start_s": round(t3 - t2, 6),
            "start_delta_mb": round(rss_after_start - rss_before_start, 3),
        },
    }

    if args.preload:
        preload_cfg = {
            "model_path": model_path,
            "max_context_size": 4096,
            "n_gpu_layers": -1,
            "max_batch_size": 512,
            "verbose": False,
        }
        if model_path and os.path.exists(model_path):
            rss_before_preload = _rss_mb()
            t4 = time.perf_counter()
            cpp_scheduler_engine.start(worker_count=1, gpu_config=preload_cfg, preload_llm=True)
            t5 = time.perf_counter()
            rss_after_preload = _rss_mb()
            asyncio.run(_stop_engine(cpp_scheduler_engine))
            payload["measurements"].update(
                {
                    "rss_before_preload_mb": round(rss_before_preload, 3),
                    "rss_after_preload_mb": round(rss_after_preload, 3),
                    "preload_s": round(t5 - t4, 6),
                    "preload_delta_mb": round(rss_after_preload - rss_before_preload, 3),
                }
            )
        else:
            payload["measurements"]["preload_skipped"] = True
            payload["measurements"]["preload_skip_reason"] = "模型文件不存在，无法 preload"

    ok = True
    if payload["measurements"]["import_s"] > float(args.import_s_max):
        ok = False
    if payload["measurements"]["start_s"] > float(args.start_s_max):
        ok = False
    if payload["measurements"]["start_delta_mb"] > float(args.start_delta_mb_max):
        ok = False
    payload["ok"] = ok

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
