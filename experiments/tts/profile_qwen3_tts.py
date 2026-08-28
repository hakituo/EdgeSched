"""连续 10 个真实 chunk 的音色 prompt 复用验收。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.tts.smoke_qwen3_tts_base import (  # noqa: E402
    build_parser,
    run_smoke,
)


def main() -> None:
    parser = build_parser()
    parser.set_defaults(chunks=10)
    asyncio.run(run_smoke(parser.parse_args()))


if __name__ == "__main__":
    main()
