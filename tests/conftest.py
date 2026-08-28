import os
import sys


def _ensure_mvp_core_on_path() -> None:
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    mvp_core_dir = os.path.dirname(tests_dir)
    legacy_dir = os.path.dirname(mvp_core_dir)
    repo_root = os.path.dirname(legacy_dir)

    for p in (repo_root, legacy_dir, mvp_core_dir):
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_mvp_core_on_path()

