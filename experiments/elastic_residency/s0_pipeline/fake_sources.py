"""S0 流水线的测试替身：只用于链路、背压与清理验证。

所有替身都把 ``backend_kind`` 标记为 ``*_fake``，产物中必须同时记录
``is_real=False``，禁止冒充真实后端或进入正式性能结论。
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import numpy as np

from experiments.elastic_residency.s0_pipeline.token_source import (
    BridgedTokenSource,
)
from experiments.elastic_residency.s0_pipeline.tts_executor import (
    SentenceSynthesizer,
)

FAKE_SAMPLE_RATE = 24000


class ScriptedTokenSource(BridgedTokenSource):
    """按脚本交付 token，可注入延迟与失败，用于验证背压与异常清理。"""

    def __init__(
        self,
        tokens: list[str],
        *,
        per_token_delay_s: float = 0.0,
        fail_after: int | None = None,
        fail_message: str = "scripted token source failure",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tokens = list(tokens)
        self.per_token_delay_s = per_token_delay_s
        self.fail_after = fail_after
        self.fail_message = fail_message
        self.produced = 0

    def _produce(self) -> Iterator[str]:
        for index, token in enumerate(self.tokens):
            if self.fail_after is not None and index >= self.fail_after:
                raise RuntimeError(self.fail_message)
            if self.per_token_delay_s > 0:
                time.sleep(self.per_token_delay_s)
            self.produced += 1
            yield token


class ScriptedSynthesizer(SentenceSynthesizer):
    """确定性假合成器：按文本长度决定时长，可注入延迟与失败。"""

    def __init__(
        self,
        *,
        synthesis_delay_s: float = 0.0,
        base_duration_s: float = 0.2,
        per_char_duration_s: float = 0.01,
        fail_on_chunk: int | None = None,
        silence_only: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=None,
            voice_prompt=None,
            language=kwargs.pop("language", "Chinese"),
            generation_kwargs=kwargs.pop("generation_kwargs", {}),
            **kwargs,
        )
        self.backend_kind = "scripted_fake"
        self.synthesis_delay_s = synthesis_delay_s
        self.base_duration_s = base_duration_s
        self.per_char_duration_s = per_char_duration_s
        self.fail_on_chunk = fail_on_chunk
        self.silence_only = silence_only

    def _generate(self, text: str) -> tuple[Any, int]:
        if self.fail_on_chunk is not None and getattr(
            self, "_current_chunk_id", None
        ) == self.fail_on_chunk:
            raise RuntimeError("scripted synthesis failure")
        if self.synthesis_delay_s > 0:
            time.sleep(self.synthesis_delay_s)
        duration_s = self.base_duration_s + len(text) * self.per_char_duration_s
        count = max(1, int(FAKE_SAMPLE_RATE * duration_s))
        timeline = np.arange(count, dtype=np.float32) / FAKE_SAMPLE_RATE
        if self.silence_only:
            # 自然静音：是真实样本，不应被判为供给不足
            samples = np.zeros(count, dtype=np.float32)
        else:
            samples = (0.05 * np.sin(2 * np.pi * 220 * timeline)).astype(
                np.float32
            )
        return [samples], FAKE_SAMPLE_RATE

    def _synthesize_blocking(self, request: Any, submit_ns: int) -> Any:
        self._current_chunk_id = request.chunk_id
        try:
            return super()._synthesize_blocking(request, submit_ns)
        finally:
            self._current_chunk_id = None
