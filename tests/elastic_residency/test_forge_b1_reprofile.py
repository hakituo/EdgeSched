"""forge_b1_reprofile 的纯逻辑测试（不依赖 psutil，不起 Forge 服务）。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from experiments.elastic_residency.forge_b1_reprofile import (
    build_txt2img_payload,
    build_parser,
    evaluate_swap_growth,
    extract_gpu_used_series,
    summarize_gpu_series,
)


def _sample(
    monotonic_ns: int,
    gpu_used_mb: float,
    *,
    gpu_available: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        monotonic_ns=monotonic_ns,
        gpu_available=gpu_available,
        gpu_card_used_mb=gpu_used_mb,
    )


def test_build_txt2img_payload_matches_batch_scaling_baseline() -> None:
    payload = build_txt2img_payload(
        seed=42,
        width=1024,
        height=1024,
        steps=8,
        cfg_scale=1.0,
        sampler="Euler",
    )
    assert payload["batch_size"] == 1
    assert payload["n_iter"] == 1
    assert payload["width"] == 1024
    assert payload["height"] == 1024
    assert payload["steps"] == 8
    assert payload["cfg_scale"] == 1.0
    assert payload["sampler_name"] == "Euler"
    assert payload["seed"] == 42
    assert payload["prompt"].startswith("a serene mountain lake")


def test_extract_gpu_used_series_filters_window_and_availability() -> None:
    samples = [
        _sample(100, 300.0),
        _sample(200, 5900.0),
        _sample(300, 5800.0, gpu_available=False),
        _sample(400, 5700.0),
        _sample(500, 400.0),
    ]
    series = extract_gpu_used_series(samples, start_ns=150, end_ns=450)
    assert series == [5900.0, 5700.0]

    assert extract_gpu_used_series(samples) == [300.0, 5900.0, 5700.0, 400.0]
    assert extract_gpu_used_series(samples, start_ns=600) == []


def test_summarize_gpu_series_handles_empty_window() -> None:
    empty = summarize_gpu_series([])
    assert empty == {"high_water_mb": None, "sample_count": 0}

    summary: dict[str, Any] = summarize_gpu_series([500.0, 5904.0, 5600.0])
    assert summary["high_water_mb"] == 5904.0
    assert summary["low_water_mb"] == 500.0
    assert summary["sample_count"] == 3


def test_parser_defaults_match_frozen_c2_baseline() -> None:
    args = build_parser().parse_args(["--run-id", "forge_test"])
    assert args.iterations == 5
    assert args.warmup == 1
    assert args.width == 1024
    assert args.height == 1024
    assert args.steps == 8
    assert args.cfg_scale == 1.0
    assert args.sampler == "Euler"
    assert args.sample_interval == 0.05
    assert args.gpu_sample_interval == 0.25
    assert args.system_cores == 2
    assert args.min_baseline_gpu_samples == 8
    assert args.max_swap_growth_mb == 4.0


def test_evaluate_swap_growth_tolerates_system_noise() -> None:
    # v1 复测观察到的 WSL2/Forge 系统噪声：780 KB 级，应在 4 MiB 容忍内。
    noise = evaluate_swap_growth([0, 0, 274432, 798720, 798720], 4 * 1024 * 1024)
    assert noise["growth_bytes"] == 798720
    assert noise["within_tolerance"] is True

    # 真实换页增长（数百 MB）必须被拒绝。
    pressure = evaluate_swap_growth([0, 512 * 1024 * 1024], 4 * 1024 * 1024)
    assert pressure["growth_bytes"] == 512 * 1024 * 1024
    assert pressure["within_tolerance"] is False

    # swap 回落不计增长；空序列按通过处理（另有样本数门禁兜底）。
    dropped = evaluate_swap_growth([798720, 0, 0], 4 * 1024 * 1024)
    assert dropped["growth_bytes"] == 0
    assert dropped["within_tolerance"] is True
    assert evaluate_swap_growth([], 1024)["within_tolerance"] is True
