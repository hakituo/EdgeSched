#!/bin/bash
export PYTHONPATH=/home/leslie:${PYTHONPATH:-}
source /home/leslie/.venvs/mvp_core-cu128/bin/activate
cd /home/leslie/mvp_core

echo "=== CUDA检查 ==="
python3 << 'EOF'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
EOF

echo ""
echo "=== 调度器导入测试 ==="
python3 -c "from services.task_scheduler import GlobalTaskScheduler; print('调度器导入成功')"
