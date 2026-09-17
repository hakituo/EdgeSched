"""S0 接入加速 TTS 后端的静态验证：后端选择、放置门禁与执行器契约。"""

import numpy as np
import pytest

from experiments.elastic_residency.s0_pipeline import s0_child, s0_parent
from experiments.elastic_residency.s0_pipeline.config import (
    S0_PLACEMENTS,
    TTS_PLACEMENT_T0,
)
from experiments.elastic_residency.s0_pipeline.tts_executor import (
    AcceleratedSentenceSynthesizer,
    SentenceSynthesizer,
)
from experiments.tts.qwen3_tts_backends import PATH_CUDA_GRAPH, PATH_UPSTREAM


def _fake_model(text: str):
    samples = np.zeros(2400, dtype=np.float32)
    return [samples], 24000


def test_accelerated_synthesizer_uses_plain_closure() -> None:
    synth = AcceleratedSentenceSynthesizer(
        model=_fake_model, voice_prompt=None, language="Chinese", generation_kwargs={}
    )
    generated, sample_rate = synth._generate("测试")
    assert sample_rate == 24000
    assert len(generated) == 1
    assert len(generated[0]) == 2400


def test_accelerated_synthesizer_accepts_bare_array() -> None:
    synth = AcceleratedSentenceSynthesizer(
        model=lambda text: (np.ones(48, dtype=np.float32), 24000),
        voice_prompt=None,
        language="Chinese",
        generation_kwargs={},
    )
    generated, _ = synth._generate("测试")
    assert len(generated[0]) == 48
    assert isinstance(synth, SentenceSynthesizer)


def test_child_defaults_to_current_integration_backend() -> None:
    args = s0_child.build_parser().parse_args(["--placement", "K24:T0"])
    assert args.tts_backend == PATH_CUDA_GRAPH
    assert args.faster_qwen3_tts_path


def test_parent_defaults_to_current_integration_backend() -> None:
    args = s0_parent.build_parser().parse_args([])
    assert args.tts_backend == PATH_CUDA_GRAPH
    assert list(args.placements) == list(S0_PLACEMENTS)


def test_frozen_placements_are_t0_only() -> None:
    assert set(S0_PLACEMENTS) == {"K24:T0", "K0:T0"}
    assert all(item.endswith(f":{TTS_PLACEMENT_T0}") for item in S0_PLACEMENTS)


def test_accelerated_backend_rejects_component_placements() -> None:
    s0_parent.validate_placements(["K24:T0", "K0:T0"], PATH_CUDA_GRAPH)
    with pytest.raises(ValueError, match="T0"):
        s0_parent.validate_placements(["K24:T3"], PATH_CUDA_GRAPH)
    with pytest.raises(ValueError, match="T0"):
        s0_parent.validate_placements(["K24:T4", "K0:T0"], PATH_CUDA_GRAPH)


def test_upstream_backend_keeps_placement_dimension() -> None:
    # 旧后端支持分部件放置，不该被新的 T0 门禁拦住
    s0_parent.validate_placements(["K24:T3", "K24:T4"], PATH_UPSTREAM)
