"""播放器 smoke 的判定测试：测量可信与连续性 SLO 必须分开。"""

from experiments.elastic_residency.s0_pipeline.events import Stage
from experiments.elastic_residency.s0_pipeline.playback_smoke import (
    evaluate_measurement,
    synthetic_sentence,
)

_BASE_STATS = {
    "is_real": True,
    "callback_count": 120,
    "consumed_samples": 144000,
    "starve_events": 0,
    "starve_samples": 0,
    "device_underflow_count": 0,
}


def _stages(submits: int, starts: int) -> dict[str, int]:
    return {
        Stage.PLAYBACK_SUBMIT.value: submits,
        Stage.PLAYBACK_START.value: starts,
    }


def _starvation(
    *,
    episodes: int = 0,
    callbacks: int = 0,
    samples: int = 0,
    in_window: int = 0,
    faults: int = 0,
    fault_samples: int = 0,
) -> dict[str, object]:
    return {
        "starve_episode_count": episodes,
        "starve_callback_count": callbacks,
        "starve_samples_total": samples,
        "supply_fault_episode_count": faults,
        "supply_fault_samples": fault_samples,
        "starve_samples_in_window": in_window,
        "window_samples": 144000,
        "starve_ratio_in_window": (
            round(in_window / 144000, 6) if in_window else 0.0
        ),
        "by_phase": {},
    }


def test_startup_wait_alone_is_not_a_supply_fault() -> None:
    verdict = evaluate_measurement(
        stats={**_BASE_STATS, "starve_events": 1},
        stages=_stages(4, 4),
        sentence_count=4,
        gap_injected=False,
        starvation=_starvation(episodes=1, callbacks=8, samples=8192),
    )
    assert verdict["measurement_ok"] is True
    # 只有启动等待，没有供给缺口 → 连续供给本身未被证伪
    assert verdict["continuity_ok"] is True


def test_measurement_ok_does_not_require_zero_faults() -> None:
    verdict = evaluate_measurement(
        stats={**_BASE_STATS, "starve_events": 2},
        stages=_stages(4, 4),
        sentence_count=4,
        gap_injected=False,
        starvation=_starvation(
            episodes=2, callbacks=25, samples=24704, faults=2, fault_samples=24704
        ),
    )
    assert verdict["measurement_ok"] is True
    assert verdict["continuity_ok"] is False
    assert verdict["continuity"]["starve_episode_count"] == 2
    # 断流次数与缺供给 callback 次数是两个单位，不能混用
    assert verdict["continuity"]["starve_callback_count"] == 25


def test_missing_events_fail_measurement_not_continuity() -> None:
    verdict = evaluate_measurement(
        stats=dict(_BASE_STATS),
        stages=_stages(4, 2),
        sentence_count=4,
        gap_injected=False,
        starvation=_starvation(),
    )
    assert verdict["measurement_ok"] is False
    assert verdict["measurement_checks"]["start_events_complete"] is False


def test_injected_gap_must_be_detected() -> None:
    verdict = evaluate_measurement(
        stats=dict(_BASE_STATS),
        stages=_stages(4, 4),
        sentence_count=4,
        gap_injected=True,
        starvation=_starvation(),
    )
    assert verdict["measurement_ok"] is False
    assert verdict["measurement_checks"]["supply_fault_detected"] is False


def test_episode_count_cannot_exceed_callback_count() -> None:
    verdict = evaluate_measurement(
        stats=dict(_BASE_STATS),
        stages=_stages(4, 4),
        sentence_count=4,
        gap_injected=False,
        starvation=_starvation(episodes=9, callbacks=3),
    )
    assert verdict["measurement_checks"]["episode_units_separated"] is False
    assert verdict["measurement_ok"] is False


def test_device_not_opened_blocks_measurement() -> None:
    verdict = evaluate_measurement(
        stats={**_BASE_STATS, "is_real": False, "callback_count": 0},
        stages=_stages(0, 0),
        sentence_count=4,
        gap_injected=False,
        starvation=_starvation(),
    )
    assert verdict["measurement_ok"] is False
    assert verdict["measurement_checks"]["device_opened"] is False


def test_synthetic_sentence_is_deterministic_and_sized() -> None:
    first = synthetic_sentence(24000, 0.5, 0)
    second = synthetic_sentence(24000, 0.5, 0)
    assert len(first) == 12000
    assert first.dtype == "float32"
    assert bool((first == second).all())
