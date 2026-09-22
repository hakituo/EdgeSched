"""验证同步与提交契约：Linux 侧拉取不得删除独有文件，Windows 侧提交必须写清改了什么。

2026-09-23 环境从 WSL2 换成本机原生 Linux 后同步方向反转：
  - Linux 侧 `scripts/sync_from_windows.sh` 从挂载的 D 盘仓库增量拉取（发起端在 Linux）
  - Windows 侧 `scripts/commit_and_push.ps1` 只做 git add/commit/push

本文件取代原先针对 `scripts/sync_to_wsl2_safe.ps1` 的 verify_wsl_sync_policy.py。
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LINUX_SYNC = PROJECT_ROOT / "scripts" / "sync_from_windows.sh"
WIN_COMMIT = PROJECT_ROOT / "scripts" / "commit_and_push.ps1"
LEGACY_SYNC = PROJECT_ROOT / "scripts" / "sync_to_wsl2_safe.ps1"
PROFILE_ENTRY = PROJECT_ROOT / "experiments" / "tts" / "profile_qwen3_tts.py"


def main() -> None:
    sync_text = LINUX_SYNC.read_text(encoding="utf-8")
    required = {
        "root_models_only": "--exclude '/models/'",
        "no_unanchored_models": "--exclude 'models/'",
        "python_cache": "--exclude '__pycache__/'",
    }
    if required["root_models_only"] not in sync_text:
        raise AssertionError("同步脚本的模型排除规则必须锚定项目根目录 /models/")
    if required["no_unanchored_models"] in sync_text:
        raise AssertionError("未锚定 models/ 会误排除 domain/models 源码")
    if required["python_cache"] not in sync_text:
        raise AssertionError("同步脚本必须排除 __pycache__")

    # 破坏性参数只允许出现在注释里（用来解释为什么禁用），不能作为实际参数传给 rsync
    code_text = "\n".join(
        line for line in sync_text.splitlines() if not line.lstrip().startswith("#")
    )
    for forbidden in ("--delete", "--mirror", "--delete-excluded"):
        if forbidden in code_text:
            raise AssertionError(
                f"同步脚本禁止把 {forbidden} 当作实际参数：Linux 侧模型与 Forge 不能被抹平"
            )

    # 提交信息规则：必须带 -Message，且拒绝无信息文案，防止历史里再出现看不懂的 commit
    commit_text = WIN_COMMIT.read_text(encoding="utf-8")
    required_message = {
        "message_param": "[string]$Message",
        "message_gate": "if (-not $Summary)",
        "meaningless_gate": "Test-MeaninglessMessage",
        "commit_file": "commit -F $MessageFile",
    }
    for name, token in required_message.items():
        if token not in commit_text:
            raise AssertionError(f"提交脚本缺少提交信息约束: {name}（{token}）")
    if "auto: 同步后自动提交" in commit_text:
        raise AssertionError("不允许回退到固定文案的自动提交")

    # WSL2 时期的同步脚本如仍留在仓库里，必须显著标明已停用
    if LEGACY_SYNC.exists():
        legacy_text = LEGACY_SYNC.read_text(encoding="utf-8")
        if "已停用" not in legacy_text:
            raise AssertionError("残留的 sync_to_wsl2_safe.ps1 必须在头部标明已停用")

    profile_text = PROFILE_ENTRY.read_text(encoding="utf-8")
    if "sys.path.insert(0, str(PROJECT_ROOT))" not in profile_text:
        raise AssertionError("TTS profile 入口必须支持直接从文件路径运行")

    updates = (PROJECT_ROOT / "docs" / "UPDATES.md").read_text(encoding="utf-8")
    if "commit 主题行" not in updates:
        raise AssertionError("docs/UPDATES.md 必须说明 commit 与详细记录的分工")
    print("[OK] 同步契约：Linux 侧拉取保留独有文件，Windows 侧提交必须写清改了什么")


if __name__ == "__main__":
    main()
