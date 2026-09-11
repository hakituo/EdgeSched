"""G3：真实 LLM 流式 token 源与有界线程桥接。

设计约束：

- 直接使用真实 ``llama_cpp``，禁止 Mock 或 C++ 调度 fallback；
- 线程到异步端的 token 缓冲必须**有界**，否则 TTS 队列虽然有界，
  背压仍然传不回 LLM；
- 记录模型身份、chat template、输入 token 数、输出上限与实际输出长度；
- 不允许先填满 context 再额外生成多句而越过冻结的 C512 口径。

时间语义：本模块只产生 **delivery** 语义的时间戳，即 token 从生成线程
交付到桥接队列的时刻，它不是模型内部逐 token 生成完成时刻。

``BridgedTokenSource`` 只负责有界桥接与统计，子类只需实现 ``_produce``，
因此测试替身与真实后端共用同一条背压路径。
"""

from __future__ import annotations

import asyncio
import hashlib
import queue as queue_module
import threading
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from experiments.elastic_residency.s0_pipeline.events import (
    ClockDomain,
    EventRecorder,
    Stage,
    TimestampSemantics,
)
from experiments.elastic_residency.s0_pipeline.s0_prompt import (
    prompt_evidence,
    render_chat_prompt,
    special_token_text,
)
from experiments.elastic_residency.s0_pipeline.token_identity import (
    TokenSourceError,
    TokenSourceIdentity,
    TokenSourceStats,
)

_END = object()


class BridgedTokenSource:
    """有界线程桥接：把同步 token 生成器变成异步流。"""

    def __init__(
        self,
        *,
        bridge_maxsize: int = 16,
        recorder: EventRecorder | None = None,
    ) -> None:
        if bridge_maxsize < 1:
            raise ValueError("token 桥接队列必须有界")
        self.bridge_maxsize = bridge_maxsize
        self.recorder = recorder
        self.stats = TokenSourceStats()
        self._bridge: Any = None
        self._thread: threading.Thread | None = None

    @property
    def bridge_depth(self) -> int:
        return int(self._bridge.qsize()) if self._bridge is not None else 0

    def _produce(self) -> Iterator[str]:
        raise NotImplementedError

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("token 源不能重复启动")
        self._bridge = queue_module.Queue(maxsize=self.bridge_maxsize)
        self._thread = threading.Thread(
            target=self._run, name="s0-token-source", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        assert self._bridge is not None
        try:
            for text in self._produce():
                if not text:
                    continue
                delivered_ns = time.monotonic_ns()
                depth_before = self._bridge.qsize()
                if self._bridge.full():
                    self.stats.bridge_blocked_count += 1
                # put 在桥接队列满时阻塞，背压因此传回生成循环
                self._bridge.put((text, delivered_ns))
                self.stats.bridge_max_depth = max(
                    self.stats.bridge_max_depth, depth_before
                )
            self.stats.finished = True
        except Exception as exc:  # noqa: BLE001 - 线程异常必须传给异步端
            self.stats.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._bridge.put(_END)

    async def stream(self) -> AsyncIterator[str]:
        if self._bridge is None:
            raise TokenSourceError("必须先 start() 再 stream()")
        loop = asyncio.get_running_loop()
        delivered = 0
        while True:
            item = await loop.run_in_executor(None, self._bridge.get)
            if item is _END:
                break
            text, delivered_ns = item
            delivered += 1
            self.stats.delivered_tokens = delivered
            self.stats.actual_output_tokens = delivered
            if self.stats.first_delivery_ns is None:
                self.stats.first_delivery_ns = delivered_ns
            self.stats.last_delivery_ns = delivered_ns
            if self.recorder is not None:
                self.recorder.emit(
                    Stage.TOKEN_DELTA,
                    clock_domain=ClockDomain.TOKEN_THREAD,
                    timestamp_ns=delivered_ns,
                    semantics=TimestampSemantics.DELIVERY,
                    attributes={
                        "token_index": delivered - 1,
                        "bridge_depth": self.bridge_depth,
                        "delivery_semantics_note": (
                            "token 交付到桥接队列的时刻，"
                            "不是模型内部生成完成时刻"
                        ),
                    },
                )
            yield text
        if self.stats.error:
            raise TokenSourceError(self.stats.error)
        if self.recorder is not None:
            self.recorder.emit(
                Stage.TOKEN_END,
                clock_domain=ClockDomain.TOKEN_THREAD,
                timestamp_ns=time.monotonic_ns(),
                semantics=TimestampSemantics.DELIVERY,
                attributes={"delivered_tokens": delivered},
            )

    def close(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None


class LlamaTokenSource(BridgedTokenSource):
    """真实 llama_cpp 后端，无任何 fallback。"""

    def __init__(
        self,
        *,
        model_path: str | Path,
        n_ctx: int,
        context_tokens: int,
        generation_budget_tokens: int,
        max_output_tokens: int,
        n_gpu_layers: int,
        n_threads: int,
        seed: int,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        model_sha256: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_path = Path(model_path).expanduser()
        self.n_ctx = n_ctx
        self.context_tokens = context_tokens
        self.generation_budget_tokens = generation_budget_tokens
        self.max_output_tokens = max_output_tokens
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.seed = seed
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.model_sha256 = model_sha256
        self.rendered_prompt = ""
        self.prompt_evidence: dict[str, Any] = {}
        self._stop: list[str] = []
        self._llm: Any = None

    def open(self) -> TokenSourceIdentity:
        """加载真实模型并校验 C512 边界；禁止任何 fallback。"""

        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - 仅真实环境命中
            raise TokenSourceError(f"缺少真实后端 llama_cpp: {exc}") from exc
        if not self.model_path.is_file():
            raise TokenSourceError(f"找不到 GGUF: {self.model_path}")
        if len(self.model_sha256) != 64:
            raise TokenSourceError("必须提供模型 SHA-256，禁止未核验身份的加载")

        offload = self.n_gpu_layers != 0
        self._llm = Llama(
            model_path=str(self.model_path),
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            n_batch=min(self.n_ctx, 512),
            n_ubatch=min(self.n_ctx, 512),
            n_threads=self.n_threads,
            n_threads_batch=self.n_threads,
            seed=self.seed,
            use_mmap=True,
            offload_kqv=offload,
            op_offload=offload,
            flash_attn=True,
            verbose=False,
        )
        metadata = dict(self._llm.metadata)
        architecture = str(metadata.get("general.architecture", ""))
        block_count = int(metadata.get(f"{architecture}.block_count", 0) or 0)
        template = str(metadata.get("tokenizer.chat_template", ""))
        eos_text = special_token_text(self._llm, eos=True)
        self.rendered_prompt = render_chat_prompt(
            chat_template=template,
            system_prompt=self.system_prompt,
            user_prompt=self.prompt,
            eos_token=eos_text,
            bos_token=special_token_text(self._llm, eos=False),
        )
        self.prompt_evidence = prompt_evidence(self.rendered_prompt)
        self._stop = [eos_text] if eos_text else []
        identity = TokenSourceIdentity(
            model_path=str(self.model_path),
            model_sha256=self.model_sha256,
            architecture=architecture,
            block_count=block_count,
            n_ctx=self.n_ctx,
            context_tokens=self.context_tokens,
            generation_budget_tokens=self.generation_budget_tokens,
            chat_template_sha256=hashlib.sha256(
                template.encode("utf-8")
            ).hexdigest(),
            prompt_tokens_estimate=self._estimate_prompt_tokens(),
            max_output_tokens=self.max_output_tokens,
            n_gpu_layers=self.n_gpu_layers,
            n_threads=self.n_threads,
        )
        if self.recorder is not None:
            self.recorder.emit(
                Stage.LLM_LOAD_END,
                clock_domain=ClockDomain.HOST_MONOTONIC,
                timestamp_ns=time.monotonic_ns(),
                semantics=TimestampSemantics.EXECUTION_END,
                attributes={"identity": identity.to_dict()},
            )
        return identity

    def _estimate_prompt_tokens(self) -> int:
        assert self._llm is not None
        return len(self._llm.tokenize(self.rendered_prompt.encode("utf-8")))

    def _produce(self) -> Iterator[str]:
        assert self._llm is not None
        stream = self._llm.create_completion(
            prompt=self.rendered_prompt,
            max_tokens=self.max_output_tokens,
            temperature=self.temperature,
            seed=self.seed,
            stream=True,
            stop=self._stop or None,
        )
        for chunk in stream:
            text = chunk["choices"][0].get("text")
            if text:
                yield str(text)

    def close(self) -> None:
        super().close()
        if self._llm is not None:
            try:
                self._llm.close()
            finally:
                self._llm = None
