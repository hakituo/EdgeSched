# 5. 实现

## 5.1 技术栈

EdgeSched 基于 Python 3.10 实现，主要依赖如下：

**核心框架**：FastAPI 0.110+ 提供 WebSocket API，Uvicorn 作为 ASGI 服务器，Pydantic 2.6+ 用于数据验证。

**机器学习推理**：llama-cpp-python 0.2+ 用于 LLM 推理（GGUF 格式），PyTorch 2.0+ 配合 Diffusers 0.25+ 用于图像生成（Stable Diffusion）。

**异步运行时**：asyncio 实现基于协程的并发，aiohttp 作为异步 HTTP 客户端。

**硬件配置**：NVIDIA GeForce RTX 5070 Laptop（8GB 显存），AMD Ryzen 9 8940HX，32GB DDR5，Windows 11。

## 5.2 关键模块实现

**动态资源迁移**。算法 1 展示了紧急卸载流程。系统首先通过 `save_state()` 保存完整推理状态（含 KV Cache、logits 与 embedding，见第 5.3 节口径说明），然后释放 GPU 显存，最后异步加载 CPU 后端。

```python
async def offload_llm_to_cpu(urgent=True):
    if urgent:
        # 保存 KV Cache
        state = await asyncio.get_running_loop().run_in_executor(
            None, old_inst.save_state
        )
        # 释放 GPU 显存
        del self._python_llm
        gc.collect()
        # 异步加载 CPU 后端
        asyncio.create_task(load_cpu_backend())
```

**任务队列管理**。图像队列采用 asyncio.Queue 实现有界容量。队列满时立即拒绝新任务，防止资源耗尽。

```python
async def try_enqueue_image_generation(prompt):
    if self._image_queue.full():
        raise RuntimeError("Image queue is full")
    await self._image_queue.put((prompt, future))
```

**状态持久化**。StateBackplane 使用 Python 的 mmap 模块实现零拷贝 I/O。任务状态通过 struct.pack 序列化后直接写入内存映射文件。

```python
def register_task(task_id, status, priority):
    offset = TASKS_OFFSET + (slot * TASK_SLOT_SIZE)
    struct.pack_into(TASK_SLOT_FORMAT, self._mmap, offset,
                     task_id.encode(), status, priority, time.time())
```

## 5.3 部署配置

**LLM 配置**：
- 模型：Qwen2.5-7B-Instruct（GGUF 格式，Q4_K_M 量化）
- 模型权重：约 4.7GB
- KV Cache：完整可迁移状态（`save_state()` 序列化，含 logits、embedding 与 KV memory）不是纯 GPU KV 显存；GPU 预分配 KV buffer 的容量由独立 KV 校准矩阵实测，正文只填写校准通过后的数值
- 总显存占用：约 6-7GB
- GPU 层数：默认 -1（自动分配），8GB 显存环境下通常加载 25-33 层
- 首字超时：30 秒
- 最大生成 token 数：512

**图像生成配置**：
- 模型：SDXL Turbo 1.0（fp16 精度，约 6.7GB）
- 加载策略：按需加载，完成后卸载以释放显存
- 最大并发数：1
- 队列容量：8
- 总槽位数：9（并发 + 排队）
- 图像尺寸：512×512
- 推理步数：20
