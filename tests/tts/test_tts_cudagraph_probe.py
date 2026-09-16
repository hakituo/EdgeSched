"""Qwen3-TTS 后端对照探针的静态验证：纯函数、参数契约与惰性导入守卫。"""

import ast
import pathlib

import numpy as np
import pytest

from experiments.elastic_residency.tts_component_offload_matrix import (
    EXPECTED_LENGTHS,
    TEXTS,
)
from experiments.tts import profile_qwen3_tts_cudagraph as probe
from experiments.tts import qwen3_tts_backends as backends

_HEAVY_MODULES = ("torch", "faster_qwen3_tts", "soundfile", "librosa")


def _module_level_imports(module_path: str) -> set[str]:
    tree = ast.parse(pathlib.Path(module_path).read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def test_heavy_backends_are_imported_lazily() -> None:
    # 探针与后端模块必须能在无 GPU / 无加速包的环境里被导入与静态验证
    for module in (probe.__file__, backends.__file__):
        modules = _module_level_imports(module)
        for name in _HEAVY_MODULES:
            assert name not in modules, f"{module} 在模块级导入了 {name}"


def test_probe_reuses_frozen_b2_texts() -> None:
    assert {name: len(text) for name, text in TEXTS.items()} == EXPECTED_LENGTHS
    assert list(TEXTS) == ["chars_20", "chars_50", "chars_100"]


def test_aggregate_reports_per_path_and_text_stats() -> None:
    rows = [
        {"path": probe.PATH_UPSTREAM, "text": "chars_20", "rtf": 1.5,
         "elapsed_s": 7.0, "audio_s": 4.7, "pcm_sha256": "a"},
        {"path": probe.PATH_UPSTREAM, "text": "chars_20", "rtf": 2.5,
         "elapsed_s": 12.0, "audio_s": 4.8, "pcm_sha256": "b"},
        {"path": probe.PATH_CUDA_GRAPH, "text": "chars_20", "rtf": 0.3,
         "elapsed_s": 1.4, "audio_s": 4.7, "pcm_sha256": "c"},
    ]
    summary = probe.aggregate(rows)
    upstream = summary[f"{probe.PATH_UPSTREAM}|chars_20"]
    assert upstream["rounds"] == 2
    assert upstream["rtf_mean"] == 2.0
    assert upstream["rtf_min"] == 1.5
    assert upstream["rtf_max"] == 2.5
    assert upstream["distinct_pcm_sha256"] == ["a", "b"]
    graph = summary[f"{probe.PATH_CUDA_GRAPH}|chars_20"]
    assert graph["rtf_mean"] == 0.3


def test_pcm_hash_is_backend_independent_and_sensitive() -> None:
    samples = np.linspace(-0.5, 0.5, 2400, dtype=np.float32)
    assert probe.pcm_sha256(samples) == probe.pcm_sha256(samples.copy())
    assert probe.pcm_sha256(samples) != probe.pcm_sha256(samples * 0.5)


def test_to_mono_accepts_list_and_array() -> None:
    samples = np.ones(10, dtype=np.float64)
    assert probe.to_mono([samples]).shape == (10,)
    assert probe.to_mono(samples).dtype == np.float32


def test_parser_defaults_run_both_paths_and_xvector() -> None:
    args = probe.build_parser().parse_args([])
    assert args.paths == [probe.PATH_UPSTREAM, probe.PATH_CUDA_GRAPH]
    assert args.rounds == 3
    assert args.xvec_only is True
    assert args.faster_qwen3_tts_path == probe.DEFAULT_FASTER_PATH


def test_parser_rejects_unknown_path() -> None:
    with pytest.raises(SystemExit):
        probe.build_parser().parse_args(["--paths", "no_such_backend"])


def test_build_synth_rejects_unknown_path() -> None:
    with pytest.raises(ValueError, match="未知推理后端路径"):
        backends.build_synth(None, "no_such_backend")
