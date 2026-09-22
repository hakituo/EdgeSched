#!/usr/bin/env bash
# =============================================================================
# EdgeSched：本机 Linux 侧增量同步（Windows 仓库 → $HOME/mvp_core）
#
# 用法（在 Linux 上运行，不需要 root）：
#   bash scripts/sync_from_windows.sh              # 增量同步
#   bash scripts/sync_from_windows.sh --dry-run    # 先看会改哪些文件
#   bash scripts/sync_from_windows.sh --ensure-links  # 顺带补齐 experiment_results 软链
#
# 背景：2026-09-23 起实验环境从 WSL2 换成本机原生 Linux（与 Windows 双系统）。
#   - 代码真源：Windows 侧 git 仓库 D:\AI\xiaoyou-core\legacy\mvp_core
#     （Linux 侧挂载为 /mnt/d/AI/xiaoyou-core/legacy/mvp_core）
#   - 运行副本：$HOME/mvp_core（放模型、编译 C++、跑实验；本身不是 git 仓库）
#   - 双系统下 Windows 与 Linux 不会同时在线，所以同步的发起端在 Linux 侧：
#     Windows 侧只管编辑与 git 提交推送（scripts/commit_and_push.ps1）。
#
# 安全语义（与 WSL2 时期一致，不要放宽）：
#   - 只增量覆盖，永不使用 --delete / --mirror：Linux 侧独有的 models、Forge、
#     experiment_results、编译产物绝不能被 Windows 侧「抹平」。
#   - 排除规则锚定项目根（'/models/' 而非 'models/'），避免误伤 domain/models 之类的源码目录。
# =============================================================================
set -uo pipefail

SOURCE="${SOURCE:-/mnt/d/AI/xiaoyou-core/legacy/mvp_core}"
DEST="${DEST:-$HOME/mvp_core}"
WIN_RESULTS="${WIN_RESULTS:-/mnt/d/AI/xiaoyou-core/legacy/mvp_core/experiment_results}"

DRY_RUN=0
ENSURE_LINKS=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|--dryrun) DRY_RUN=1; shift ;;
        --ensure-links) ENSURE_LINKS=1; shift ;;
        --source) SOURCE="$2"; shift 2 ;;
        --dest) DEST="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "未知参数：$1（用 --help 查看）"; exit 2 ;;
    esac
done

die() { echo "[错误] $*" >&2; exit 1; }

[[ -d "$SOURCE" ]] || die "找不到 Windows 仓库：$SOURCE
  通常是没挂载 D 盘。先执行（设备名用 lsblk -f 确认）：
    sudo mkdir -p /mnt/d && sudo mount -t ntfs3 /dev/nvme1n1p4 /mnt/d"
[[ -f "$SOURCE/main.py" ]] || die "$SOURCE 看起来不是 mvp_core 仓库（缺 main.py）"
command -v rsync >/dev/null 2>&1 || die "缺少 rsync：sudo apt-get install -y rsync"

mkdir -p "$DEST"

echo "源（Windows）：$SOURCE/"
echo "目标（Linux）：$DEST/"
echo "策略：增量覆盖；删除 Linux 侧独有文件；排除根锚定的 models/Forge/results/缓存。"
echo ""

RSYNC_ARGS=(
    -rtl --info=stats2
    --exclude '/models/'
    --exclude '/experiment_results/'
    --exclude '/output/'
    --exclude '/stable-diffusion-webui-forge-main/'
    --exclude '/stable-diffusion-webui-forge-main.zip'
    --exclude '/cpp_scheduler/build/'
    --exclude '/cpp_scheduler/build_py310/'
    --exclude '.git/'
    --exclude '__pycache__/'
    --exclude '*.pyc'
    --exclude '.pytest_cache/'
    --exclude '.ruff_cache/'
    --exclude '.test_tmp*/'
)
[[ $DRY_RUN -eq 1 ]] && RSYNC_ARGS+=(--dry-run -i)

rsync "${RSYNC_ARGS[@]}" "$SOURCE/" "$DEST/" || die "rsync 失败"
[[ $DRY_RUN -eq 1 ]] && { echo; echo "dry-run 结束，未改动任何文件。"; exit 0; }

if [[ $ENSURE_LINKS -eq 1 ]]; then
    if [[ -d "$WIN_RESULTS" ]]; then
        if [[ -L "$DEST/experiment_results" || ! -e "$DEST/experiment_results" ]]; then
            ln -sfn "$WIN_RESULTS" "$DEST/experiment_results"
            echo "[OK] experiment_results -> $WIN_RESULTS"
        else
            echo "[注意] $DEST/experiment_results 是真实目录，未改动"
        fi
    else
        echo "[注意] 未找到 Windows 侧结果目录 $WIN_RESULTS，跳过软链"
    fi
fi

echo ""
echo "完成。Linux 侧运行环境固定在："
echo "  代码   $DEST"
echo "  主环境 \$HOME/.venvs/mvp_core-cu128（软链到加速 TTS 环境）"
echo "  模型   $DEST/models"
