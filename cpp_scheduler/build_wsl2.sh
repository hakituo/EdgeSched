#!/bin/bash
set -euo pipefail

# 激活虚拟环境（稳定路径 mvp_core-cu128 现为软链；
# 2026-09-25 起指向 /home/leslie/xiaoyou-core/.venv，Python 3.13 + torch 2.12+cu130）
source /home/leslie/.venvs/mvp_core-cu128/bin/activate

# CUDA 12.8（RTX 5070 Laptop = Blackwell sm_120，需要 12.8+）
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
PYBIND11_CMAKE_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')"

SCRIPT_DIR="/home/leslie/mvp_core/cpp_scheduler"
BUILD_DIR="${SCRIPT_DIR}/build"
PYTHON_EXE="$(which python)"

echo "=== Phase-Aware C++ Scheduler Build ==="
echo "Python: $(python --version) at ${PYTHON_EXE}"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'not found')"
echo "cmake: $(cmake --version | head -1)"
echo ""

# 清理旧 build
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "=== Running CMake ==="
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_LLAMA_CUDA=ON \
    -DCUDAToolkit_ROOT=/usr/local/cuda \
    -DCMAKE_CUDA_ARCHITECTURES=120 \
    -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}" \
    -DPython_EXECUTABLE="${PYTHON_EXE}" \
    -DPYTHON_EXECUTABLE="${PYTHON_EXE}"

echo ""
echo "=== Building ==="
cmake --build . -j$(nproc)

echo ""
echo "=== Build Complete ==="
ls -la scheduler_py*.so 2>/dev/null || echo "WARNING: scheduler_py not found"
