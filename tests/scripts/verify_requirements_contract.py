"""验证依赖契约：Linux 实验环境真源与实际环境一致，Forge 隔离，旧 WSL2 清单已标记失效。

背景：2026-09-17「软链换名」后 requirements-wsl-cu128.txt 描述的主环境已被搁置，
实际环境中 transformers / torch / numpy 等都与该文件不符；2026-09-23 迁移到本机 Linux 时
以实际环境 pip freeze 为准重写了真源，本测试锁定该契约。

本文件取代原先的 verify_wsl_requirements.py。
"""

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
    linux_requirements = _read("requirements-linux-cu128.txt")
    forge_requirements = _read("requirements-linux-forge.txt")
    legacy_wsl = _read("requirements-wsl-cu128.txt")
    environment = _read("environment.yml")

    required_general = {
        "accelerate==1.12.0",
        "safetensors==0.8.0",
        "llama-cpp-python==0.3.35",
        # 实际发行名是 qwen-tts-hf；写成 qwen-tts 会装到另一个包
        "qwen-tts-hf==0.1.1.post1",
    }
    required_linux = {
        "--extra-index-url https://download.pytorch.org/whl/cu128",
        "torch==2.11.0+cu128",
        "torchaudio==2.11.0+cu128",
        "transformers==5.15.1",
        "numpy==2.5.3",
        "onnxruntime==1.30.0",
        "ctranslate2==4.8.2",
    }
    required_forge = {
        "torch==2.10.0+cu128",
        "torchvision==0.25.0+cu128",
        "numpy==1.26.2",
        "gradio==4.40.0",
    }

    for contract in required_general:
        if contract not in requirements:
            raise AssertionError(f"requirements.txt 缺少: {contract}")
    for contract in required_linux:
        if contract not in linux_requirements:
            raise AssertionError(f"requirements-linux-cu128.txt 缺少: {contract}")
    for contract in required_forge:
        if contract not in forge_requirements:
            raise AssertionError(f"requirements-linux-forge.txt 缺少: {contract}")

    # Forge 与主环境必须保持隔离：两个环境不能收敛到同一套 torch / numpy
    if "numpy==1.26.2" in linux_requirements:
        raise AssertionError("主环境清单不应出现 Forge 的 numpy 1.26.2（两个环境必须隔离）")

    if "-r requirements-linux-cu128.txt" not in environment:
        raise AssertionError("environment.yml 未指向当前依赖真源 requirements-linux-cu128.txt")
    if "pytorch-cuda=12.1" in environment:
        raise AssertionError("environment.yml 仍包含过时 CUDA 12.1 契约")

    # 旧 WSL2 清单必须显著标记失效，避免有人照它重建环境
    if "已失效" not in legacy_wsl:
        raise AssertionError("requirements-wsl-cu128.txt 必须标明已失效")
    if "requirements-linux-cu128.txt" not in legacy_wsl:
        raise AssertionError("requirements-wsl-cu128.txt 必须指向新的依赖真源")

    print("[OK] Linux 依赖契约一致，Forge 与主环境隔离，旧 WSL2 清单已标记失效")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
