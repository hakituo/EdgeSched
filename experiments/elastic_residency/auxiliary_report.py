"""辅助核数矩阵计划与有效性/SLA 分离判定。"""

from __future__ import annotations

import random
import statistics
import math

from .resource_sampler import _percentile

CORE_COUNTS = {"json_rag": (1, 2), "bge": (1, 2, 4), "bge_cached": (1, 2, 4),
               "whisper": (1, 2, 4)}
SLA = {"json_rag": 0.050, "bge": 0.150, "bge_cached": 0.150, "whisper": 0.8}


def cached_evidence_valid(row: dict) -> bool:
    """缓存画像必须显式预计算且请求阶段只编码 query；旧模式不能冒充新模式。"""
    prep = row.get("preparation", {})
    elapsed = prep.get("elapsed_s")
    cache_hash = prep.get("cache_sha256", "")
    identity = prep.get("candidate_identity_sha256", "")
    return bool(
        prep.get("kind") == "candidate_precompute" and prep.get("candidate_count") == 32
        and prep.get("embedding_shape") == [32, 512]
        and prep.get("included_in_request_latency") is False
        and prep.get("query_cache_enabled") is False
        and isinstance(elapsed, (float, int)) and math.isfinite(elapsed) and elapsed >= 0
        and len(cache_hash) == 64 and len(identity) == 64 and row.get("events")
        and all(e.get("output", {}).get("candidates_reencoded") is False
                and e["output"].get("embedding_shape") == [1, 512]
                and e["output"].get("candidate_embedding_shape") == [32, 512]
                and e["output"].get("backend") == "onnxruntime_cpu_cached_candidate_rerank"
                and e["output"].get("query_cache_enabled") is False
                and e["output"].get("cache_sha256") == cache_hash
                and e["output"].get("candidate_identity_sha256") == identity
                for e in row["events"]))


def build_plan(tasks: list[str], reserves: list[int], repeats: int, seed: int) -> list[dict]:
    if (repeats < 1 or not tasks or len(set(tasks)) != len(tasks)
            or any(t not in CORE_COUNTS for t in tasks)
            or not reserves or len(set(reserves)) != len(reserves)
            or any(r not in (0, 1, 2, 4) for r in reserves)):
        raise ValueError("任务、重复次数或系统预留核数不合法")
    rng = random.Random(seed)
    plan = []
    for repeat in range(repeats):
        block = [{"task": task, "cores": cores, "system_cores": reserve,
                  "repeat": repeat + 1} for reserve in reserves for task in tasks
                 for cores in CORE_COUNTS[task]]
        rng.shuffle(block)
        plan.extend(block)
    return plan


def summarize(plan: list[dict], results: list[dict], events_per_run: int) -> dict:
    groups = []
    keys = sorted({(p["task"], p["system_cores"], p["cores"]) for p in plan})
    for task, reserve, cores in keys:
        def matches(row):
            return (row["task"], row["system_cores"], row["cores"]) == (task, reserve, cores)
        expected = [p for p in plan if matches(p)]
        rows = [r for r in results if matches(r)]
        valid = (len(rows) == len(expected)
                 and sorted(r["repeat"] for r in rows) == sorted(p["repeat"] for p in expected)
                 and all(r.get("measurement_valid") and len(r.get("events", []))
                         == events_per_run
                         and (task != "bge_cached" or cached_evidence_valid(r)) for r in rows))
        values = [event["rtf" if task == "whisper" else "elapsed_s"]
                  for r in rows if r.get("measurement_valid") for event in r["events"]]
        enough = valid and len(rows) >= 5 and len(values) >= 30
        p95 = _percentile(values, .95) if enough else None
        groups.append({"task": task, "system_cores": reserve, "cores": cores,
                       "expected_runs": len(expected), "completed_runs": len(rows),
                       "measurement_valid": valid, "event_count": len(values),
                       "mean": statistics.fmean(values) if values else None,
                       "min": min(values) if values else None,
                       "max": max(values) if values else None, "p95": p95,
                       "unit": "rtf" if task == "whisper" else "seconds",
                       "sla": SLA[task], "sla_passed": p95 <= SLA[task] if enough else None,
                       "status": ("invalid" if not valid else "insufficient_samples"
                                  if not enough else "feasible" if p95 <= SLA[task]
                                  else "infeasible")})
        if task == "bge_cached":
            costs = [r["preparation"]["elapsed_s"] for r in rows if cached_evidence_valid(r)]
            groups[-1]["candidate_precompute_s"] = {
                "sample_count": len(costs), "mean": statistics.fmean(costs) if costs else None,
                "min": min(costs) if costs else None, "max": max(costs) if costs else None}
    recommendations = []
    for task, reserve in sorted({(g["task"], g["system_cores"]) for g in groups}):
        candidates = [g for g in groups if (g["task"], g["system_cores"]) == (task, reserve)]
        complete = (set(g["cores"] for g in candidates) == set(CORE_COUNTS[task])
                    and all(g["status"] in ("feasible", "infeasible") for g in candidates))
        feasible = [g["cores"] for g in candidates if g["sla_passed"] is True]
        recommendations.append({"task": task, "system_cores": reserve,
                                "minimum_cores": min(feasible) if complete and feasible else None,
                                "status": "incomplete" if not complete else
                                "feasible" if feasible else "infeasible"})
    return {"evidence_kind": "real_cpu_isolated_reservation_sensitivity",
            "expected_runs": len(plan), "completed_runs": len(results),
            "measurement_valid": len(results) == len(plan) and all(
                g["measurement_valid"] for g in groups),
            "groups": groups, "minimum_core_budgets": recommendations,
            "formal_r_sys": {"status": "not_evaluated", "passed": None,
                             "missing": ["concurrent_fixed_llm_tts_pipeline",
                                         "paired_no_auxiliary_baseline",
                                         "real_chunk_playback_underrun",
                                         "decode_ttfa_reply_loss_le_5_percent"]},
            "paper_promoted": False}
