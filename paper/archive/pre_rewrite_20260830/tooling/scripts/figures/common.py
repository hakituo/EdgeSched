import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "experiment_data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "figures"

FIG_MIGRATION = OUTPUT_DIR / "01_paper_main" / "migration"
FIG_ROBUSTNESS = OUTPUT_DIR / "01_paper_main" / "robustness"
FIG_BASELINE = OUTPUT_DIR / "01_paper_main" / "baseline"
FIG_AUXILIARY = OUTPUT_DIR / "02_auxiliary"


def save(name: str, subdir: Path):
    subdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(subdir / f"{name}.pdf", bbox_inches='tight', pad_inches=0.15)
    plt.savefig(subdir / f"{name}.png", bbox_inches='tight', pad_inches=0.15)
    print(f"Saved: {subdir / name}.pdf")
    plt.close()


def load_json(filename):
    for subdir in ["01_paper_main", "01_paper_main/migration", "02_auxiliary"]:
        # 尝试带 .json 后缀
        filepath = RESULTS_DIR / subdir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        # 尝试不带后缀（run_all_experiments.py 保存的格式）
        filepath_no_ext = RESULTS_DIR / subdir / filename.replace('.json', '')
        if filepath_no_ext.exists():
            with open(filepath_no_ext, 'r', encoding='utf-8') as f:
                return json.load(f)
    filepath = RESULTS_DIR / filename
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None
