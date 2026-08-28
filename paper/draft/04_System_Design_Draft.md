# 4. System Design

## 4.1 Architecture Overview

图 1 展示了 EdgeSched 的整体架构。系统采用分层设计，自上而下分为表示层、领域层、控制面、执行层和数据适配层。用户请求通过 WebSocket 接口进入系统，经领域层业务逻辑处理后，由控制面统一路由到相应的执行组件。

控制面包含三个核心组件：
（1）GlobalTaskScheduler 作为 L1 路由层，负责任务分发，管理优先级队列、取消事件和图像任务背压控制；
（2）CPPSchedulerEngine 作为 L2 高性能执行层，封装 llama-cpp-python 推理引擎，实现 GPU/CPU 动态资源迁移；
（3）StateBackplane 基于 mmap 实现零拷贝状态持久化，支持崩溃后 12ms 内恢复任务状态。

执行层包含 ResourceManager，负责协调模型生命周期和 GPU/CPU 资源分配，仲裁多模型间的资源竞争。数据适配层包含 LLM Adapter 和 Image Adapter，分别封装大语言模型（GGUF 格式）和图像生成模型（Stable Diffusion）的推理逻辑，屏蔽底层推理引擎的差异，支持多种后端无缝切换。

## 4.2 Dual-Task Scheduling

EdgeSched 针对边缘设备上最常见的两类大模型应用——对话交互（LLM）与图像生成（Diffusion）——设计双任务调度机制。LLM 属于常驻交互任务（Resident），需要持续响应用户输入，优先常驻 GPU 以保障低延迟；图像生成属于一次性任务（One-shot），计算密集且在 CPU 上延迟不可接受，工程上必须依赖 GPU。

系统采用不同的调度策略处理两类任务。LLM 任务通过 `submit_llm_task()` 直接提交至 CPPSchedulerEngine，默认占用 GPU 资源；图像生成任务通过 `try_enqueue_image_generation()` 进入独立的有界队列。当图像任务到达时，若 LLM 占用 GPU 显存，系统立即触发动态资源迁移：保存 KV Cache 后释放显存，供图像任务加载执行。若多个图像任务连续到达，后续任务在队列中排队，队列满时触发背压机制拒绝新任务，避免资源耗尽导致系统崩溃。

这种设计确保常驻任务（LLM）的持续可用性，同时为突发任务（图像生成）提供可预测的资源分配策略。

## 4.3 Dynamic Resource Migration

边缘设备配备独立显卡时，显存与系统内存通过 PCIe 总线分离。LLM 常驻 GPU（含 KV Cache 约 6-7GB）可保障低延迟交互，但 8GB 显存无法同时容纳 LLM 与图像生成模型（约 6-7GB）。当图像生成任务到达时，必须立即释放 LLM 占用的显存，否则触发 CUDA OOM。此时显存成为硬约束，用户体验退化为软约束——短暂的服务降级优于系统崩溃。反之，当图像任务完成、显存空闲时，将 LLM 迁回 GPU 恢复高性能，无需牺牲上下文连续性。这种约束条件的非对称性，决定了下行路径（GPU→CPU）采用紧急模式优先释放资源，上行路径（CPU→GPU）采用 Atomic Hot-Swap（原子热切换）保障服务可用。

### 4.3.1 Asymmetric Migration Strategy

**Downlink (GPU → CPU): Urgent Mode**. 当图像任务到达时，系统触发紧急迁移以立即释放显存。该模式首先通过 `save_state()` 将 KV Cache 从显存经 PCIe 复制到内存（约 200-300ms），保留对话上下文；随后立即释放 GPU 显存，消除显存碎片以避免 OOM 风险。释放完成后，系统并行启动两个操作：后台加载 CPU 版 LLM（从 DDR 到 DDR，利用 Page Cache 加速，>20GB/s），以及加载图像生成模型到 GPU（从 DDR 经 PCIe 到 GDDR）。此模式通过短暂的上下文保存延迟，换取显存安全与用户体验的兼顾。

**Uplink (CPU → GPU): Atomic Hot-Swap**. 当图像任务完成、显存空闲时，系统执行原子热切换以恢复 LLM 的高性能推理。该模式采用 Double Buffering 策略：
1. **Preload**: 首先在 GPU 上全新加载 LLM 实例（从 DDR 经 PCIe 到 GDDR），此时 CPU 实例仍在服务用户请求（不中断）；
2. **Hot Transfer**: 待 GPU 实例就绪后，立即暂停 CPU 实例，通过 `save_state()` 提取 KV Cache 并注入 GPU 实例（纯内存操作）；
3. **Switch & Unload**: 最后原子替换实例指针，将新请求路由到 GPU 实例，并异步卸载 CPU 实例。

此模式将服务中断时间从秒级压缩到毫秒级（仅状态注入耗时），最大化服务可用性。

### 4.3.2 Implementation Mechanisms

非对称迁移的实现依赖三个核心机制：
1. **Concurrency Control**: `asyncio.Lock()` 保证实例替换的原子性，避免竞态条件；
2. **State Serialization**: llama-cpp-python 提供的 `save_state()` / `load_state()` 接口实现 KV Cache 的跨设备迁移，状态数据经 PCIe 在显存与内存间传输；
3. **Async Cleanup**: 通过 `asyncio.create_task()` 后台释放旧实例，避免阻塞服务。

## 4.4 Backpressure Control

图像生成任务计算密集且显存占用大，系统采用有界队列防止资源耗尽。图像队列容量由 `queue_capacity`（默认 8）和 `max_concurrent`（默认 1）共同决定，总槽位数为两者之和（默认 9）。其中 `max_concurrent` 表示并发执行数，`queue_capacity` 表示排队等待数。

队列管理通过 `try_enqueue_image_generation()` 实现：系统使用 `asyncio.Queue(maxsize=total_slots)` 创建有界队列，若队列未满，任务入队等待执行；若队列已满，立即抛出 `RuntimeError("Image queue is full")` 拒绝请求。这种设计确保系统在高压场景下保持可预测的行为，而非无限堆积任务导致内存耗尽或响应延迟失控。

此外，系统跟踪图像任务状态（pending/running/completed），支持任务取消和结果查询，为用户提供透明的任务生命周期管理。

## 4.5 Cancellation Semantics

系统支持用户主动取消正在执行或排队的任务，采用协作式中断策略避免资源泄漏。

**LLM 流式取消**。LLM 生成采用流式输出，通过 `asyncio.Event` 实现协作式取消。用户调用 `cancel_llm_request(request_id)` 后，系统设置取消标志；生成循环在每个 token 后检查标志，若标志已设置则立即停止后续生成。此模式避免强制终止导致的显存泄漏，确保资源安全释放。

**图像任务取消**。排队中的图像任务直接丢弃；正在执行的任务设置取消标志，但无法强制中断（需等待完成）。系统通过 `cancel_image_job(job_id)` 接口提供取消能力，已取消任务的结果将被丢弃，不向用户推送。

**局限性**。系统不支持硬中断（强制杀死 GPU 任务），这是由 CUDA 显存管理机制决定的。正在运行的图像任务必须等待完成，取消仅影响结果是否推送。这种设计承认不可硬中断的边界，在资源安全与响应速度之间取得平衡。

## 4.6 Crash Recovery

边缘设备环境不稳定，系统可能因断电或异常崩溃。StateBackplane 基于 mmap 实现零拷贝状态持久化，设计目标是在系统重启后 12ms 内恢复任务状态。

**数据布局**。持久化文件大小为 1MB，分为三个区域：Header（8B）存储 Magic Number 和版本号，用于验证数据完整性；Task Slots（32个槽位，每个 48B）存储任务 ID、状态、优先级和时间戳；剩余空间预留扩展。

**持久化机制**。任务状态变更时，通过 `register_task()` 和 `update_task_status()` 实时写入 mmap 文件。由于 mmap 将文件直接映射到内存地址空间，读写操作等价于内存访问，无需传统 I/O 的数据拷贝，延迟仅为微秒级。

**恢复流程**。系统重启时，StateBackplane 初始化读取持久化文件：验证 Magic Number 和版本号确保数据格式正确；遍历 32 个任务槽位，提取状态非零的活跃任务；将状态为 Pending 的任务重新加入调度队列；将状态为 Running 的任务标记为 Failed，避免悬挂任务。该设计确保崩溃后任务状态不丢失，提供最小可用的任务闭合语义。
