# 实验工作产物

该目录保存运行结果，不等于论文正文数据。

## 当前仍使用

- `elastic_residency/`：当前弹性驻留实验；内部历史批次已按自己的 `_archive` 分类。
- `tts/`：当前 Qwen3-TTS smoke 和 profiling。
- `batch_scaling/`：Forge batch scaling 探索性结果。

## 已归档

- `_archive/pre_rewrite_20260830/legacy_paper_v1/`：上一版论文使用或生成的 baseline、Mock Phase-Aware、压力、恢复和旧 streaming 结果。
- `_archive/pre_rewrite_20260830/duplicate_root_TTS/`：原根目录 `TTS/` 的重复副本；内容哈希与 `experiment_results/tts/` 中对应文件一致。
- `_archive/pre_rewrite_20260830/local_logs/`：整理时存在的本地日志。

新结果先进入对应实验目录。只有人工核验后，才允许晋升到 `paper/experiment_data/`。
