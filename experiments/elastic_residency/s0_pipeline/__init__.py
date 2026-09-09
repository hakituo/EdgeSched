"""S0 无图像 LLM 到 TTS 端到端流水线实验包。

本包只补四个容量可行配置的真实交互证据：LLM 流式输出、固定分句、
有界队列、逐句合成、连续播放，并输出可解释的统一时钟事件流。

首轮交付目标是「真实依赖链与可解释时间线通过验收」，不是性能 Go，
也不是 planner 开工许可。模块依赖顺序：

    config 到 events 到 token_source / tts_executor / playback 到 runner
"""

from __future__ import annotations
