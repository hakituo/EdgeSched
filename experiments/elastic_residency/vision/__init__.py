"""Qwen3.5-4B + mmproj 多模态端点门禁子包。

模块职责（与 `AGENTS.md` 的 300 行上限约束一致，按 artifact / 运行后端 /
测量证据 / 实验编排四类拆分）：

artifact
- `gguf_reader.py`：GGUF 头部元数据只读解析，不加载模型；
- `vision_artifacts.py`：主 GGUF 与 mmproj 的画像、哈希与对应关系核验；
- `fixed_image.py`：确定性固定测试图片（像素内容哈希与编码器无关）；
- `vision_endpoint_gate.py`：任何真实测量前必须通过的 artifact/能力门禁。

运行后端
- `mtmd_runtime.py`：libmtmd projector warm、bitmap、tokenize 与释放；
- `mtmd_eval.py`：chunk 级真实执行、状态同步与固定长度解码；
- `vision_endpoint_phases.py`：文本端点与视觉端点的真实执行阶段。

测量证据
- `vision_evidence.py`：证据语义、驻留状态、prompt 构造与视觉门禁原语；
- `vision_endpoint_checks.py`：单轮硬门禁装配；
- `vision_endpoint_contracts.py`：run_config 与 iteration 记录装配。

实验编排
- `vision_endpoint_cli.py`：命令行契约与子进程命令构造；
- `vision_endpoint_child.py`：单轮真实端点测量子进程；
- `vision_endpoint_report.py`：单元级聚合与 CSV 汇总；
- `vision_endpoint_smoke.py`：K0/FULL × 文本/视觉端点 smoke 入口。

证据边界：本子包只提供真实执行路径与门禁。Mock 只用于静态测试链路，
不产生任何性能或可行性结论；projector 未实际执行的轮次必须失败，
禁止静默退回纯文本，也禁止把 mmproj 文件大小当作运行期显存。
"""
