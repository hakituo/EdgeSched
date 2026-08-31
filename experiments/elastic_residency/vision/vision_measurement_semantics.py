"""多模态端点测量字段的语义字典。

每个字段都必须能在产物里反向定位，并说明它**不是**什么：这里记录的边界
（不是文件大小、不是预留量、不是可复现性门禁）与字段含义同等重要。
"""

from __future__ import annotations


VISION_MEASUREMENT_SEMANTICS: dict[str, str] = {
    "residency_state": (
        "本轮的驻留状态：TEXT_ONLY / TEXT_ONLY_CHAT 表示 projector 未加载；"
        "VISION_WARM_GPU / VISION_WARM_CPU 表示 projector 已 warm 且对固定图片"
        "实际执行。状态由代码路径决定，不允许事后改名。"
    ),
    "visual_token_total": (
        "libmtmd 报告的图像 chunk token 总数（mtmd_input_chunk_get_n_tokens），"
        "是真实进入语言模型的视觉输入规模，不是图片像素数或文件大小。"
    ),
    "visual_position_total": (
        "图像 chunk 在语言模型上下文中实际推进的位置数。启用 M-RoPE 时它小于"
        "visual_token_total（实测 14×14=196 个视觉 token 塌缩为 14 个位置），"
        "两者分别记录，不互相替代。"
    ),
    "projector_encode_s": (
        "split_encode 模式下 mtmd_encode_chunk 的独立耗时，即 projector"
        "（视觉编码器 + merger）真实执行时间；chunk_single 模式下为 None。"
    ),
    "visual_embedding_available": (
        "split_encode 模式下 mtmd_get_output_embd 是否返回非空视觉嵌入；"
        "这是 projector 真实执行的直接证据。chunk_single 模式下为 None，"
        "此时只以返回码与位置推进作为间接证据。"
    ),
    "projector_load_s": "mtmd_init_from_file 的真实耗时，即 projector warm 成本。",
    "projector_free_s": "mtmd_free 的真实耗时，即 projector 释放成本。",
    "projector_gpu_card_delta_mb": (
        "projector warm 前后 direct NVML 整卡显存差值；隔离环境下近似为 projector"
        "的显存驻留增量，不是精确进程显存，也不是 mmproj 文件大小。"
        "实测该差值在同一环境下有约 ±80 MiB 波动（CUDA 分配器碎片与整卡其他"
        "进程共同影响），因此必须多轮取统计，禁止用单轮读数当作常量。"
    ),
    "projector_rss_delta_bytes": (
        "projector warm 前后进程 RSS 差值；use_gpu=False 时主要体现在这里。"
    ),
    "uses_non_causal_encode": (
        "该图像 chunk 的编码是否使用非因果注意力，由 mtmd_decode_use_non_causal"
        "(ctx, chunk) 按 chunk 查询，不是 context 级能力，因此逐 chunk 记录。"
    ),
    "projector_warmup_enabled": (
        "libmtmd 的 warmup 是否开启。开启时会按 mmproj 声明的最大像素预留 compute"
        "buffer（与本轮图片尺寸无关），该预留计入 projector warm 的显存增量；解释"
        "M_V 时必须说明这一部分是最大尺寸预留，而非本轮图片的实际编码开销。"
    ),
    "projector_warm_noise_note": (
        "projector warm 的 GPU 增量包含三部分：mmproj 权重（约 644.26 MiB）、"
        "libmtmd warmup 按 mmproj 声明的最大像素预留的 compute buffer（本模型"
        "为 1472×1472、223.30 MiB，与本轮图片尺寸无关）、以及 CUDA 上下文与"
        "分配器开销。要拿「本轮图片真实编码开销」必须另外限制 image_max_pixels"
        "或关闭 warmup 再测，不能直接引用该差值。"
    ),
    "prompt_eval_s": (
        "视觉轮次的 prefill 语义：全部文本与图像 chunk 的真实执行总时长，"
        "包含 projector 编码与视觉嵌入 decode，不能与纯文本 prefill 直接比较。"
    ),
    "kv_tokens_used_is_position_count": (
        "kv_native_state.kv_tokens_used 记录的是 KV 序列**位置数**，不是原始"
        "token 数。视觉轮次启用 M-RoPE 时两者不同：196 个视觉 token 塌缩为 14"
        "个序列位置，因此 KV 位置总数为 文本 token + 视觉位置 + 生成 token。"
        "跨文本/视觉比较 KV 占用时必须说明使用的是哪一种口径。"
    ),
    "kv_sequence_state_bytes_span_note": (
        "kv_sequence_state_bytes 由**实际使用的位置跨度**决定，不是 n_ctx 的"
        "预留量：512+128 纯文本轮次为 73,676,076 B，而视觉轮次因位置跨度更小"
        "（167 位置）只有 64,134,768 B。它描述已写入的 KV，不是 n_ctx 比例化的"
        "驻留成本，两者不可互相替代。"
    ),
    "output_text_reproducibility_note": (
        "output_text 及其 SHA-256 是内容证据，不是可复现性门禁：纯文本轮次在"
        "同一 seed 下逐字节一致，但视觉轮次受 GPU kernel 累加顺序非确定性影响，"
        "同一 seed 多次运行措辞可能不同。视觉轮次应以「语义是否正确描述固定"
        "图片」判定，禁止要求输出哈希逐字节复现。"
    ),
    "image_file_sha256": "固定输入图片的文件 SHA-256。",
    "image_pixel_payload_sha256": (
        "固定图片的原始 RGB 像素哈希；与 PNG 编码器版本无关，是图片内容身份。"
    ),
    "text_only_prompt_semantics": (
        "TEXT_ONLY 使用与 C1/C2 相同的固定 token 数填充 prompt；"
        "TEXT_ONLY_CHAT 使用与视觉轮次相同的对话 prompt 但不含图片。"
        "跨状态比较 TTFT 时必须说明使用的是哪一种对照。"
    ),
}
