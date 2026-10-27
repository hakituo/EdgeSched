# EdgeSched 实验环境：用 `source scripts/experiments/edgesched_env.sh` 载入。
#
# 主环境是 Autarkia 的共享 venv（/home/leslie/.venvs/mvp_core-cu128 软链指向它）。
# llama-cpp-python 的 CUDA 版需要 libcudart.so.12 / libcublas.so.12，这些库由 venv 里的
# nvidia-*-cu12 pip 包提供。系统 /etc/ld.so.conf.d/xiaoyou-nvidia-cu12.conf 仍指向已不存在的
# xiaoyou-core 路径，因此这里用 LD_LIBRARY_PATH 补上，不需要 root。

EDGESCHED_VENV="${EDGESCHED_VENV:-/home/leslie/.venvs/mvp_core-cu128}"
EDGESCHED_ROOT="${EDGESCHED_ROOT:-/home/leslie/mvp_core}"

_edgesched_nvidia_libs=""
for _dir in "$EDGESCHED_VENV"/lib/python3.*/site-packages/nvidia/*/lib; do
    [ -d "$_dir" ] && _edgesched_nvidia_libs="${_edgesched_nvidia_libs:+$_edgesched_nvidia_libs:}$_dir"
done

export LD_LIBRARY_PATH="${_edgesched_nvidia_libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$EDGESCHED_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export EDGESCHED_PYTHON="$EDGESCHED_VENV/bin/python"
# TTS 权重：父项目已从 xiaoyou-core 改名为 Autarkia
export EDGESCHED_TTS_MODEL="${EDGESCHED_TTS_MODEL:-/home/leslie/Autarkia/models/Qwen3-TTS-12Hz-0.6B-Base}"
unset _dir _edgesched_nvidia_libs
