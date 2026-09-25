import json
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np


base_dir = os.path.dirname(os.path.abspath(__file__))


def _default_results_dir() -> str:
    return os.path.join(base_dir, "..", "experiment_results")


def _try_load(path: str) -> dict | None:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_result(results_dir: str, mode: str, workload: str) -> dict | None:
    if workload == "real":
        return _try_load(os.path.join(results_dir, f"{mode}_real.json"))
    if workload == "mock":
        direct = os.path.join(results_dir, f"{mode}_mock.json")
        payload = _try_load(direct)
        if payload is not None:
            return payload
        nested = os.path.join(results_dir, "verify_performance", f"{mode}_mock.json")
        return _try_load(nested)
    return None


def _load_backpressure(results_dir: str, workload: str) -> dict | None:
    direct = os.path.join(results_dir, f"image_backpressure_{workload}.json")
    payload = _try_load(direct)
    if payload is not None:
        return payload
    nested = os.path.join(results_dir, "verify_performance", f"image_backpressure_{workload}.json")
    return _try_load(nested)


def _get_exp1_rps_at_concurrency(result: dict, concurrency: int) -> float:
    exp1 = result.get("results", {}).get("exp1", [])
    for row in exp1:
        if int(row.get("concurrency", -1)) == int(concurrency):
            return float(row.get("rps", 0.0))
    if not exp1:
        return 0.0
    best = max(exp1, key=lambda r: int(r.get("concurrency", 0)))
    return float(best.get("rps", 0.0))


def _extract_exp1_rows(result: dict) -> list[dict]:
    exp1 = result.get("results", {}).get("exp1", [])
    if isinstance(exp1, list):
        return exp1
    return []


def _get_exp2_lag_metrics_ms(result: dict) -> dict:
    m = result.get("results", {}).get("exp2", {})
    return {
        "avg_lag_ms": float(m.get("avg_lag", 0.0)),
        "max_lag_ms": float(m.get("max_lag", 0.0)),
    }


def _get_exp3_metrics_seconds(result: dict) -> dict:
    m = result.get("results", {}).get("exp3_metrics", {})
    return {
        "p50_s": float(m.get("p50_ms", 0.0)) / 1000.0,
        "p95_s": float(m.get("p95_ms", 0.0)) / 1000.0,
        "p99_s": float(m.get("p99_ms", 0.0)) / 1000.0,
    }


def _plot_throughput_comparison(series: list[tuple[str, dict]], title: str, out_dir: str, out_prefix: str, fallback_concurrency: int) -> None:
    if not series:
        return

    conc_values: list[int] = []
    for _, r in series:
        for row in _extract_exp1_rows(r):
            try:
                conc_values.append(int(row.get("concurrency", 0)))
            except Exception:
                pass
    conc_values = sorted({c for c in conc_values if c > 0})
    if not conc_values:
        conc_values = [int(fallback_concurrency)]

    plt.figure(figsize=(10, 6))
    for label, r in series:
        rps = [_get_exp1_rps_at_concurrency(r, c) for c in conc_values]
        plt.plot(conc_values, rps, marker="o", linewidth=2.2, label=label)

    plt.xlabel("Concurrency")
    plt.ylabel("Throughput (RPS)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    out_pdf = os.path.join(out_dir, f"{out_prefix}_throughput_comparison.pdf")
    out_png = os.path.join(out_dir, f"{out_prefix}_throughput_comparison.png")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)
    print(f"Saved {out_pdf}")
    plt.close()


def _plot_latency_percentiles(series: list[tuple[str, dict]], title: str, out_dir: str, out_prefix: str) -> None:
    labels: list[str] = []
    p50: list[float] = []
    p95: list[float] = []
    p99: list[float] = []

    for label, r in series:
        if r is None:
            continue
        labels.append(label)
        m = _get_exp3_metrics_seconds(r)
        p50.append(m["p50_s"])
        p95.append(m["p95_s"])
        p99.append(m["p99_s"])

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    r1 = ax.bar(x - width, p50, width, label="P50")
    r2 = ax.bar(x, p95, width, label="P95")
    r3 = ax.bar(x + width, p99, width, label="P99")

    ax.set_ylabel("Latency (seconds)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(r1, padding=3, fmt="%.2f")
    ax.bar_label(r2, padding=3, fmt="%.2f")
    ax.bar_label(r3, padding=3, fmt="%.2f")

    out_pdf = os.path.join(out_dir, f"{out_prefix}_latency_percentiles.pdf")
    out_png = os.path.join(out_dir, f"{out_prefix}_latency_percentiles.png")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)
    print(f"Saved {out_pdf}")
    plt.close(fig)


def _plot_event_loop_lag(series: list[tuple[str, dict]], title: str, out_dir: str, out_prefix: str) -> None:
    labels: list[str] = []
    avg_lag: list[float] = []
    max_lag: list[float] = []

    for label, r in series:
        if r is None:
            continue
        labels.append(label)
        m = _get_exp2_lag_metrics_ms(r)
        avg_lag.append(m["avg_lag_ms"])
        max_lag.append(m["max_lag_ms"])

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    r1 = ax.bar(x - width / 2, avg_lag, width, label="Avg Lag")
    r2 = ax.bar(x + width / 2, max_lag, width, label="Max Lag")

    ax.set_ylabel("Event Loop Lag (ms)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.bar_label(r1, padding=3, fmt="%.1f")
    ax.bar_label(r2, padding=3, fmt="%.1f")

    out_pdf = os.path.join(out_dir, f"{out_prefix}_event_loop_lag.pdf")
    out_png = os.path.join(out_dir, f"{out_prefix}_event_loop_lag.png")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)
    print(f"Saved {out_pdf}")
    plt.close(fig)


def _plot_image_backpressure(bp: dict | None, title: str, out_dir: str, out_prefix: str) -> None:
    if bp is None:
        return

    res = bp.get("results", {}) if isinstance(bp, dict) else {}
    accepted = float(res.get("accepted", 0))
    rejected = float(res.get("rejected", 0))
    errors = float(res.get("errors", 0))
    finished = float(res.get("finished_accepted", res.get("finished_total", res.get("finished_count", 0))))

    labels = ["Accepted", "Rejected", "Errors", "Finished"]
    values = [accepted, rejected, errors, finished]

    plt.figure(figsize=(9.5, 5.5))
    bars = plt.bar(labels, values, color=["#2ca02c", "#d62728", "#9467bd", "#1f77b4"])
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        height = float(bar.get_height())
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    out_pdf = os.path.join(out_dir, f"{out_prefix}_image_backpressure.pdf")
    out_png = os.path.join(out_dir, f"{out_prefix}_image_backpressure.png")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)
    print(f"Saved {out_pdf}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", default="real", choices=["real", "mock"])
    parser.add_argument("--results_dir", default=_default_results_dir())
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--fallback_concurrency", type=int, default=10)
    args = parser.parse_args()

    results_dir = os.path.abspath(str(args.results_dir))
    out_dir = os.path.abspath(str(args.out_dir)) if args.out_dir else results_dir
    os.makedirs(out_dir, exist_ok=True)

    workload = str(args.workload)
    xy = _load_result(results_dir, "xy_core", workload)
    xy_cpp = _load_result(results_dir, "xy_core_cpp_sched", workload)

    series: list[tuple[str, dict]] = []
    if xy is not None:
        series.append(("EdgeSched (Python)", xy))
    if xy_cpp is not None:
        series.append(("EdgeSched (C++ Hybrid)", xy_cpp))

    if not series:
        print(f"Warning: No {workload} workload result files found in: {results_dir}")
    else:
        title_prefix = "Real" if workload == "real" else "Mock"
        _plot_throughput_comparison(
            series,
            title=f"{title_prefix} Workload Throughput vs Concurrency",
            out_dir=out_dir,
            out_prefix=workload,
            fallback_concurrency=int(args.fallback_concurrency),
        )
        _plot_latency_percentiles(
            series,
            title=f"{title_prefix} Workload Latency Percentiles",
            out_dir=out_dir,
            out_prefix=workload,
        )
        _plot_event_loop_lag(
            series,
            title=f"{title_prefix} Workload Event Loop Lag (Exp2)",
            out_dir=out_dir,
            out_prefix=workload,
        )

    bp = _load_backpressure(results_dir, workload)
    if bp is None:
        print(f"Warning: Missing image_backpressure_{workload}.json under: {results_dir}")
    else:
        title_prefix = "Real" if workload == "real" else "Mock"
        _plot_image_backpressure(
            bp,
            title=f"Image Generation Backpressure Microbench ({title_prefix})",
            out_dir=out_dir,
            out_prefix=workload,
        )
