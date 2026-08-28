"""验证 EdgeSched WSL2/CUDA 实验依赖契约。"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        raise AssertionError(f"缺少依赖文件: {relative_path}")
    return path.read_text(encoding="utf-8")


def main() -> int:
    requirements = _read("requirements.txt")
    wsl_requirements = _read("requirements-wsl-cu128.txt")
    environment = _read("environment.yml")

    required_general = {
        "transformers==4.57.3",
        "accelerate==1.12.0",
        "safetensors==0.8.0",
        "llama-cpp-python==0.3.35",
        "qwen-tts==0.1.1",
    }
    required_wsl = {
        "--no-binary llama-cpp-python",
        "torch==2.10.0+cu128",
        "torchvision==0.25.0+cu128",
        "torchaudio==2.10.0+cu128",
        "CMAKE_ARGS=\"-DGGML_CUDA=on\"",
        "sox libsox-fmt-all",
        "-r requirements.txt",
    }

    for contract in required_general:
        if contract not in requirements:
            raise AssertionError(f"requirements.txt 缺少: {contract}")
    for contract in required_wsl:
        if contract not in wsl_requirements:
            raise AssertionError(f"requirements-wsl-cu128.txt 缺少: {contract}")
    if "-r requirements-wsl-cu128.txt" not in environment:
        raise AssertionError("environment.yml 未复用 WSL2 依赖真源")
    if "pytorch-cuda=12.1" in environment:
        raise AssertionError("environment.yml 仍包含过时 CUDA 12.1 契约")

    print("[OK] EdgeSched WSL2/CUDA 依赖契约一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
