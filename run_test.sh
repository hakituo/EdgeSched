#!/bin/bash
export PYTHONPATH=/home/leslie:$PYTHONPATH
source /home/leslie/.venvs/mvp_core-cu128/bin/activate
cd /home/leslie/mvp_core

echo "=== 运行调度器开销验证 ==="
python3 tests/verify_scheduler_overhead.py 2>&1 | head -50
