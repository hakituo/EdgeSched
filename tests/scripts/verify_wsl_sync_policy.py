"""验证 Windows→WSL 同步不会误排除源码中的 models 包，且自动提交必须写清改了什么。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "sync_to_wsl2_safe.ps1"
PROFILE_ENTRY = PROJECT_ROOT / "experiments" / "tts" / "profile_qwen3_tts.py"


def main() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    required = {
        "root_models_only": "--exclude '/models/'",
        "no_unanchored_models": "--exclude 'models/'",
        "skip_git_switch": "[switch]$SkipGit",
        "python_cache": "--exclude '__pycache__/'",
    }
    if required["root_models_only"] not in text:
        raise AssertionError("模型排除规则必须锚定项目根目录 /models/")
    if required["no_unanchored_models"] in text:
        raise AssertionError("未锚定 models/ 会误排除 domain/models 源码")
    if required["skip_git_switch"] not in text:
        raise AssertionError("同步脚本缺少 -SkipGit 安全开关")
    if required["python_cache"] not in text:
        raise AssertionError("同步脚本必须排除 __pycache__")

    # 提交信息规则：必须带 -Message，且拒绝无信息文案，防止历史里再出现看不懂的 commit
    required_message = {
        "message_param": "[string]$Message",
        "message_gate": "if (-not $Summary)",
        "meaningless_gate": "Test-MeaninglessMessage",
        "commit_file": "commit -F $MessageFile",
    }
    for name, token in required_message.items():
        if token not in text:
            raise AssertionError(f"同步脚本缺少提交信息约束: {name}（{token}）")
    if "auto: 同步后自动提交" in text:
        raise AssertionError("不允许回退到固定文案的自动提交")

    profile_text = PROFILE_ENTRY.read_text(encoding="utf-8")
    if "sys.path.insert(0, str(PROJECT_ROOT))" not in profile_text:
        raise AssertionError("TTS profile 入口必须支持直接从文件路径运行")

    updates = (PROJECT_ROOT / "docs" / "UPDATES.md").read_text(encoding="utf-8")
    if "commit 主题行" not in updates:
        raise AssertionError("docs/UPDATES.md 必须说明 commit 与详细记录的分工")
    print("[OK] WSL 同步规则保留 domain/models，可跳过 Git 自动提交，且提交必须写清改了什么")


if __name__ == "__main__":
    main()
