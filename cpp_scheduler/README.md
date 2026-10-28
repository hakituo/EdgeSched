# EdgeSched C++ Scheduler

本目录是 EdgeSched 的原生执行面：C++17 资源隔离调度器、worker、任务队列、监控、llama.cpp 模型实现、Phase-Aware 策略和 pybind11 绑定。它由 Python 控制面通过 `services/cpp_scheduler_wrapper.py` 与 `services/cpp_scheduler_engine.py` 调用，不提供独立 REST 服务。

## 当前结构

| 路径 | 职责 |
|---|---|
| `core/resource_isolation_scheduler.*` | 任务类型、优先级、状态、队列分发、worker 选择和迁移回调 |
| `core/async_scheduler.*` | 旧异步调度兼容路径 |
| `core/native_executor.*` | 原生执行器 |
| `core/nvram_binding.*` | mmap/状态背板原生绑定 |
| `core/phase_strategy.*` | Phase-Aware 属性、lease、代价决策和策略 |
| `queue/task_queue.*` | 任务队列 |
| `workers/gpu_llm_worker.*` | GPU LLM worker 与任务循环 |
| `workers/llama_model_impl.*` | llama.cpp 加载、推理、会话 KV cache 保存/恢复 |
| `workers/worker_base.*` | worker 基础实现 |
| `monitoring/` | 资源监控和 flight recorder |
| `config/` | C++ 配置模型与校验 |
| `bindings/python_bindings.cpp` | `scheduler_py` Python API |
| `CMakeLists.txt` | 构建定义 |
| `build_wsl2.sh` | WSL2 推荐构建入口 |

## 与 Python 的边界

```text
GlobalTaskScheduler
  -> CPPSchedulerEngine
  -> cpp_scheduler_wrapper
  -> scheduler_py (pybind11)
  -> ResourceIsolationScheduler / LLM worker / phase strategy
```

- Python 负责请求编排、准入、取消、图像队列、状态闭合和实验采样。
- C++ 负责原生任务队列、worker、llama.cpp 推理和可选 Phase-Aware 策略。
- 绑定可导入只证明接口存在；真实 GGUF、CUDA、KV cache 和迁移仍需单独验证。

## 本机 Linux 构建（脚本名沿用 `build_wsl2.sh`）

推荐在 `/home/leslie/mvp_core` 中运行：

```bash
bash cpp_scheduler/build_wsl2.sh
```

脚本使用 `/home/leslie/.venvs/mvp_core-cu128`（软链到父项目 `/home/leslie/Autarkia/.venv` 的 Python 3.13），配置 CMake 后构建 `scheduler_py`；pybind11 取自该 venv（`pybind11.get_cmake_dir()`），CUDA 架构锁定 `CMAKE_CUDA_ARCHITECTURES=120`（RTX 5070 Laptop）。构建前确认 CUDA 12.8、CMake、编译器和 Python 环境与目标实验一致；首次构建需要系统包 `libuv1-dev`。

直接导入既有构建产物时，需要让动态链接器能找到 llama.cpp：

```bash
export LD_LIBRARY_PATH="/home/leslie/mvp_core/cpp_scheduler/build/bin:/home/leslie/mvp_core/cpp_scheduler/build/llama.cpp/src:${LD_LIBRARY_PATH:-}"
```

2026-09-25 已在本机原生 Linux 的 Python 3.13 共享 venv + CUDA 12.8 / sm_120 下用该脚本全量重编通过，产物为 `build/scheduler_py.cpython-313-x86_64-linux-gnu.so`（旧的 cpython-310 产物随 `build/` 清理已不存在）；把 `cpp_scheduler/build` 加入 `PYTHONPATH` 后 `scheduler_py` 可导入，`llama_cpp` 也可导入。组合加载时注意 `build/bin` 的 `libggml*` 可能覆盖 `llama_cpp` 自带库。导入通过不等同于真实 GGUF/CUDA 推理已经通过。

`build/` 包含第三方依赖源码和本机构建产物，体积很大；它不是源码真源，不应参与普通全文搜索、文档索引或提交。

## 验证层级

```bash
# 绑定 API 静态/导入验证
python cpp_scheduler/verify_bindings.py

# 最小启动与真实模型测试（需要模型和构建产物）
python tests/test_cpp_scheduler_start_only.py
python tests/test_cpp_scheduler_minimal.py
```

Phase-Aware Python 语义可先用：

```bash
python -m pytest tests/test_phase_aware_quick.py -q
```

真实 C++ 结论至少要分别确认：扩展导入、调度启动、真实 GGUF 生成、取消、KV cache 保存/恢复、GPU→CPU、CPU→GPU 和进程退出清理。

## 状态与谨慎表述

- `core/nvram_binding.*` 和 Python `StateBackplane` 使用文件映射模拟持久内存语义，不等同于真实 PMEM/NVRAM 硬件。
- Phase-Aware 同时存在 Python 与 C++ 实现，结构或枚举变更必须检查绑定和实验脚本。
- 历史 `docs/` 中的 API/部署说明可能包含已删除的 server、worker 或配置；当前实现以本 README、[项目上下文](../docs/PROJECT_CONTEXT.md) 和源码为准。
