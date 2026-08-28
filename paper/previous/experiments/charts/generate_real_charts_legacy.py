import json
import os

import matplotlib.pyplot as plt
import numpy as np


base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "..", "experiment_results")
os.makedirs(results_dir, exist_ok=True)


def _load_real_result(mode: str) -> dict | None:
    file_path = os.path.join(results_dir, f"{mode}_real.json")
    return _try_load(file_path)


def _try_load(path: str) -> dict | None:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


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


def plot_real_throughput_comparison(concurrency: int = 10) -> None:
    xy = _load_real_result("xy_core")
    xy_cpp = _load_real_result("xy_core_cpp_sched")
    
    series: list[tuple[str, dict]] = []
    if xy is not None:
        series.append(("xy-core (Python)", xy))
    if xy_cpp is not None:
        series.append(("xy-core (C++ Hybrid)", xy_cpp))

    if not series:
        print("Warning: No real workload result files found")
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
        conc_values = [concurrency]

    plt.figure(figsize=(10, 6))
    for label, r in series:
        rps = [_get_exp1_rps_at_concurrency(r, c) for c in conc_values]
        plt.plot(conc_values, rps, marker="o", linewidth=2.2, label=label)

    plt.xlabel("Concurrency")
    plt.ylabel("Throughput (RPS)")
    plt.title("Real Workload Throughput vs Concurrency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    out_pdf = os.path.join(results_dir, "real_throughput_comparison.pdf")
    out_png = os.path.join(results_dir, "real_throughput_comparison.png")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)
    print(f"Saved {out_pdf}")
    plt.close()


def plot_real_latency_percentiles() -> None:
    xy = _load_real_result("xy_core")
    xy_cpp = _load_real_result("xy_core_cpp_sched")

    labels: list[str] = []
    p50: list[float] = []
    p95: list[float] = []
    p99: list[float] = []

    if xy is not None:
        labels.append("xy-core (Python)")
        m = _get_exp3_metrics_seconds(xy)
        p50.append(m["p50_s"])
        p95.append(m["p95_s"])
        p99.append(m["p99_s"])

    if xy_cpp is not None:
        labels.append("xy-core (C++ Hybrid)")
        m = _get_exp3_metrics_seconds(xy_cpp)
        p50.append(m["p50_s"])
        p95.append(m["p95_s"])
        p99.append(m["p99_s"])

    if not labels:
        print("Warning: No real workload result files found")
        return

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    r1 = ax.bar(x - width, p50, width, label="P50")
    r2 = ax.bar(x, p95, width, label="P95")
    r3 = ax.bar(x + width, p99, width, label="P99")

    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Real Workload Latency Percentiles")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(r1, padding=3, fmt="%.1f")
    ax.bar_label(r2, padding=3, fmt="%.1f")
    ax.bar_label(r3, padding=3, fmt="%.1f")

    out_pdf = os.path.join(results_dir, "real_latency_percentiles.pdf")
    out_png = os.path.join(results_dir, "real_latency_percentiles.png")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)
    print(f"Saved {out_pdf}")
    plt.close()


def plot_real_event_loop_lag() -> None:
    xy = _load_real_result("xy_core")
    xy_cpp = _load_real_result("xy_core_cpp_sched")

    labels: list[str] = []
    avg_lag: list[float] = []
    max_lag: list[float] = []

    if xy is not None:
        labels.append("xy-core (Python)")
        m = _get_exp2_lag_metrics_ms(xy)
        avg_lag.append(m["avg_lag_ms"])
        max_lag.append(m["max_lag_ms"])

    if xy_cpp is not None:
        labels.append("xy-core (C++ Hybrid)")
        m = _get_exp2_lag_metrics_ms(xy_cpp)
        avg_lag.append(m["avg_lag_ms"])
        max_lag.append(m["max_lag_ms"])

    if not labels:
        print("Warning: No real workload result files found")
        return

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    r1 = ax.bar(x - width / 2, avg_lag, width, label="Avg Lag")
    r2 = ax.bar(x + width / 2, max_lag, width, label="Max Lag")

    ax.set_ylabel("Event Loop Lag (ms)")
    ax.set_title("Real Workload Event Loop Lag (Exp2)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.bar_label(r1, padding=3, fmt="%.1f")
    ax.bar_label(r2, padding=3, fmt="%.1f")

    out_pdf = os.path.join(results_dir, "real_event_loop_lag.pdf")
    out_png = os.path.join(results_dir, "real_event_loop_lag.png")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)
    print(f"Saved {out_pdf}")
    plt.close(fig)


def plot_image_backpressure_real() -> None:
    bp = _try_load(os.path.join(results_dir, "image_backpressure_real.json"))
    
    if bp is None:
        print("Warning: Missing image_backpressure_real.json")
        return

    res = bp.get("results", {}) if isinstance(bp, dict) else {}
    accepted = float(res.get("accepted", 0))
    rejected = float(res.get("rejected", 0))
    errors = float(res.get("errors", 0))
    # 使用 finished_accepted
    finished = float(res.get("finished_accepted", res.get("finished_count", 0)))

    labels = ["Accepted", "Rejected", "Errors", "Finished"]
    values = [accepted, rejected, errors, finished]

    plt.figure(figsize=(9.5, 5.5))
    bars = plt.bar(labels, values, color=["#2ca02c", "#d62728", "#9467bd", "#1f77b4"])
    plt.ylabel("Count")
    plt.title("Image Generation Backpressure Microbench (Real)")
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

    out_pdf = os.path.join(results_dir, "image_backpressure_real.pdf")
    out_png = os.path.join(results_dir, "image_backpressure_real.png")
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)
    print(f"Saved {out_pdf}")
    plt.close()


if __name__ == "__main__":
    plot_real_throughput_comparison(concurrency=10)
    plot_real_latency_percentiles()
    plot_real_event_loop_lag()
    plot_image_backpressure_real()
