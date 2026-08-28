"""验证 EdgeSched 项目索引、Codex 入口与关键路径的一致性。"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[2]

DOCS_TO_CHECK = (
    ROOT / "AGENTS.md",
    ROOT / "README.md",
    ROOT / "docs" / "PROJECT_INDEX.md",
    ROOT / "docs" / "PROJECT_CONTEXT.md",
    ROOT / "experiments" / "README.md",
    ROOT / "paper" / "README.md",
    ROOT / "cpp_scheduler" / "README.md",
)

REQUIRED_PATHS = (
    ROOT / ".trae" / "skills" / "xiaoyou-mvp-core" / "SKILL.md",
    ROOT / "services" / "task_scheduler.py",
    ROOT / "services" / "admission_controller.py",
    ROOT / "services" / "resource_manager.py",
    ROOT / "services" / "state_backplane.py",
    ROOT / "services" / "phase_aware" / "phase_scheduler.py",
    ROOT / "cpp_scheduler" / "core" / "phase_strategy.cpp",
    ROOT / "experiments" / "run_all_experiments.py",
    ROOT / "experiments" / "run_no_forge_experiments.py",
    ROOT / "experiments" / "run_forge_experiments.py",
    ROOT / "paper" / "experiment_data" / "01_paper_main",
)

MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def _resolve_local_link(source: Path, raw_target: str) -> Path | None:
    """解析 Markdown 本地链接；网络地址与页内锚点不参与检查。"""
    target = raw_target.strip().strip("<>")
    if not target or target.startswith(("#", "http://", "https://", "mailto:")):
        return None
    if target.startswith("file:"):
        raise AssertionError(f"禁止 file URI: {source.relative_to(ROOT)} -> {target}")
    path_part = unquote(target.split("#", 1)[0])
    return (source.parent / path_part).resolve()


def verify_required_paths() -> None:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_PATHS if not path.exists()]
    assert not missing, f"关键路径不存在: {missing}"


def verify_markdown_links() -> None:
    for document in DOCS_TO_CHECK:
        assert document.is_file(), f"入口文档不存在: {document.relative_to(ROOT)}"
        text = document.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            resolved = _resolve_local_link(document, raw_target)
            if resolved is not None:
                assert resolved.exists(), (
                    f"失效链接: {document.relative_to(ROOT)} -> {raw_target}"
                )


def verify_codex_entry() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert len(agents.encode("utf-8")) < 32 * 1024, "局部 AGENTS.md 超过默认 32 KiB 限制"
    for required in (
        "docs/PROJECT_INDEX.md",
        "docs/PROJECT_CONTEXT.md",
        ".trae/skills/xiaoyou-mvp-core/SKILL.md",
        "paper/experiment_data/01_paper_main/",
    ):
        assert required in agents, f"AGENTS.md 缺少入口: {required}"


def verify_context_contract() -> None:
    context = (ROOT / "docs" / "PROJECT_CONTEXT.md").read_text(encoding="utf-8")
    for fact in (
        "EdgeSched",
        "GlobalTaskScheduler",
        "PhaseAwareScheduler",
        "/home/leslie/mvp_core",
        "paper/experiment_data/01_paper_main/",
        "父仓库 `.gitignore` 含 `/legacy/`",
    ):
        assert fact in context, f"PROJECT_CONTEXT.md 缺少关键事实: {fact}"


def main() -> None:
    verify_required_paths()
    verify_markdown_links()
    verify_codex_entry()
    verify_context_contract()
    print("[OK] EdgeSched 项目索引、Codex 入口和关键路径一致")


if __name__ == "__main__":
    main()

