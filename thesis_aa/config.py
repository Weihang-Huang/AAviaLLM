"""Central configuration for the thesis_aa package.

==========================================================================
  WHAT THIS FILE IS (for newcomers)
==========================================================================

This is the "control panel" for the whole package. It defines:

  * **Paths** — where models, data, logs, and results live on disk. All paths
    are derived from the repository root (the parent of ``thesis_aa/``), so
    the package works no matter where you clone the repo. The directories are
    created automatically on import so first-run is friction-free.

  * **GPT-2 model sizes** — the four variants the thesis evaluates
    (base/medium/large/xl). The key is a short label; the value is the
    HuggingFace hub name plus a human-readable parameter count. The default
    for CPU debugging is ``"gpt2"`` (base, 124M) because it fits in memory
    and runs in seconds on CPU.

  * **Vocabulary size** — GPT-2's BPE vocabulary is 50,257 tokens. This
    constant is used everywhere in LERF / LERF-AA to size arrays.

  * **ALMs training hyperparameters** — copied from Ch.5 Table 2 of the
    thesis (100 epochs, lr 2e-5, etc.). The ``fp16`` flag here is a static
    default for CPU debugging; ``alms/train.py`` auto-detects CUDA and
    enables fp16 on GPU runs regardless of this flag.

  * **MFW feature-set sizes** — the seven vocabulary-subset sizes LERF-AA
    evaluates (Ch.7 Sec 7.3.2): 50, 100, 150, 200, 500, 1000, and the full
    50,257-type vocabulary.

  * **Device selection** — ``get_device()`` picks CUDA if available, then
    xpu (Intel GPU), then CPU. You can override by passing ``prefer="cpu"``.

All defaults favour CPU debugging so the pipelines run in seconds on a
laptop. Override values via function arguments when you move to real runs.
"""

from __future__ import annotations

import os
import torch

# Repository root (two levels up from this file: thesis_aa/config.py -> Thesis/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, "data")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")
NATURAL_DIR = os.path.join(DATA_DIR, "natural")
BENCHMARK_DIR = os.path.join(DATA_DIR, "benchmarks")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
LOG_DIR = os.path.join(ROOT_DIR, "log")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

for _d in (DATA_DIR, SYNTHETIC_DIR, NATURAL_DIR, BENCHMARK_DIR, MODEL_DIR,
           LOG_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

# GPT-2 variants evaluated in the thesis (Ch.5 Table 2, Ch.6, Ch.7).
# Each value is (huggingface_hub_name, param_count_str).
MODEL_SIZES = {
    "base": ("gpt2", "124M"),
    "medium": ("gpt2-medium", "355M"),
    "large": ("gpt2-large", "774M"),
    "xl": ("gpt2-xl", "1.5B"),
}

# GPT-2 vocabulary size (BPE), used throughout LERF / LERF-AA.
GPT2_VOCAB_SIZE = 50257

# Default base model used for ALMs further-pretraining (Ch.5 Sec 5.3.4).
DEFAULT_BASE_MODEL = "gpt2"

# Further-pretraining hyperparameters (Ch.5 Table 2).
ALMS_TRAIN_CONFIG = {
    "epochs": 100,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 64,
    "block_size": 128,
    "fp16": False,  # set True if a CUDA/xpu device is available
}

# MFW feature-set sizes evaluated in LERF-AA (Ch.7 Sec 7.3.2).
MFW_SIZES = [50, 100, 150, 200, 500, 1000, GPT2_VOCAB_SIZE]

# Standard relative-frequency estimators compared against LERF (Ch.6 Sec 6.3.1).
STANDARD_ESTIMATORS = [
    "MLE",
    "Add-One",
    "Good-Turing",
    "Katz-Backoff",
    "Kneser-Ney",
    "Witten-Bell",
]

# Benchmark datasets used in the thesis (Ch.4 Sec 4.1).
BENCHMARKS = ["Blogs50", "CCAT50", "Guardian", "IMDB62"]

# Repo URL for downloading benchmark subsets.
REPO_DATA_URL = "https://raw.githubusercontent.com/Weihang-Huang/ALMs/main/data"


def get_device(prefer: str | None = None) -> torch.device:
    """Return a torch device.

    Priority: explicit ``prefer`` > cuda/xpu if available > cpu.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")