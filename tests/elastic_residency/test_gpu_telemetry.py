from experiments.elastic_residency.gpu_telemetry import DirectNvmlProbe


class _FakeBindings:
    def device_metrics(self):
        return {
            "card_used_mb": 512.5,
            "card_total_mb": 8151.0,
            "utilization_percent": 7.0,
            "power_w": 42.5,
            "temperature_c": 55.0,
        }

    def process_memory(self, _pid: int):
        return None, "driver_reported_non_numeric_process_memory"


def test_direct_nvml_probe_reports_backend_and_device_metrics() -> None:
    result = DirectNvmlProbe(bindings=_FakeBindings()).sample(123)

    assert result["available"] is True
    assert result["backend"] == "direct_nvml"
    assert result["card_used_mb"] == 512.5
    assert result["process_memory_queried"] is False
    assert result["process_memory_available"] is False
    assert result["process_memory_unavailable_reason"] == "process_query_disabled"


def test_direct_nvml_probe_preserves_unavailable_process_memory() -> None:
    result = DirectNvmlProbe(
        include_process_memory=True,
        bindings=_FakeBindings(),
    ).sample(123)

    assert result["available"] is True
    assert result["process_memory_queried"] is True
    assert result["process_used_mb"] is None
    assert (
        result["process_memory_unavailable_reason"]
        == "driver_reported_non_numeric_process_memory"
    )
