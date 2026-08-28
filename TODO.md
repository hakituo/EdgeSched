# 系统改进 TODO 列表

## 高优先级（影响论文实验）

### 1. 紧急模式保存 KV Cache
**状态**：✅ 已完成（2026-02-25）
**原现状**：`offload_llm_to_cpu(urgent=True)` 直接释放显存，未保存 KV Cache
**已实现**：紧急模式下先 `save_state()` 保存 KV Cache，再释放显存，并在后台 CPU 实例加载后 `load_state()` 恢复
**影响**：需要重跑实验 1（资源迁移性能测试）
**代码位置**：`services/cpp_scheduler_engine.py` 的 `offload_llm_to_cpu(urgent=True)` 分支

### 2. 崩溃恢复逻辑
**状态**：✅ 已完成（2026-02-25）
**原现状**：StateBackplane 只实现状态持久化，系统重启后没有恢复任务
**已实现**：系统启动时读取 StateBackplane，标记 Running 为 Failed，并将 Pending 任务重新入队闭合
**影响**：需要新增实验 2（崩溃恢复测试）
**代码位置**：`services/task_scheduler.py` 的启动阶段恢复逻辑（读取 `get_active_tasks()`）

### 3. 统一实验参数
**状态**：✅ 已完成（2026-02-25）
**原现状**：不同实验使用不同参数（LLM: 32/50/200 tokens，图像: 5/20/25 steps）
**已实现**：统一使用标准参数
- LLM: max_tokens=64, temperature=0.7
- 图像: steps=20, 512×512, guidance_scale=7.5
**影响**：需要用新口径重跑实验并更新结果
**代码位置**：`experiments/` 目录下各实验脚本

## 中优先级（优化）

### 4. PythonWorker 清理
**状态**：✅ 已完成（2026-02-25）
**已实现**：停止时取消未完成 Future、清空残留任务/队列，并更干净地关闭线程池，避免挂起与资源泄漏
**影响**：不影响论文，稳定性更好

### 5. TaskType 清理
**状态**：✅ 已完成（2026-02-25）
**已实现**：保留 TaskType，并通过 `schedule_cpu_task()` / `schedule_gpu_task()` 作为明确入口；增加校验脚本覆盖该语义
**影响**：不影响论文，接口更清晰

## 低优先级（未来工作）

### 6. Bio State 分离
**状态**：✅ 已完成（2026-02-25）
**已实现**：直接移除 Bio State 持久化相关代码，StateBackplane 仅负责任务状态（论文主线不涉及）
**影响**：不影响论文，模块边界更清晰

### 7. 通用任务调度
**状态**：✅ 已完成（2026-02-25）
**已实现**：`schedule_task/schedule_cpu_task/schedule_gpu_task` 作为统一入口，并补齐验证脚本确保语义正确
**影响**：论文不涉及

## 需要重跑的实验
- 实验 1：资源迁移性能（紧急模式加 save_state 后）
- 实验 2：崩溃恢复（新增恢复逻辑后）
- 所有实验：统一参数后重跑

## 需要新增的实验
- 实验 2：崩溃恢复测试（系统重启后任务恢复）

## 下一步（建议优先级）
1. **重跑实验 1（真实模型）**：在 WSL2 用统一参数重新跑 `experiments/migration/resource_migration_experiment.py` 并更新 `experiment_results`
2. **新增/补齐实验 2（崩溃恢复）**：增加“系统启动自动恢复”的验证项，覆盖：
   - Running → Failed 闭合数量正确
   - Pending 任务可被重新入队并最终闭合（成功/失败均可，但不能永远挂起）
