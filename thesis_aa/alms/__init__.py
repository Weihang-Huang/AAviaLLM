"""ALMs subpackage: Authorial Language Models (Ch.5)."""

from .train import train_authorial_model, train_all_authors
from .ppl import (
    compute_ce_per_text,
    score_all_pairs,
    aggregate_ppl,
    predict_and_benchmark,
    compute_cnll,
)

__all__ = [
    "train_authorial_model",
    "train_all_authors",
    "compute_ce_per_text",
    "score_all_pairs",
    "aggregate_ppl",
    "predict_and_benchmark",
    "compute_cnll",
]