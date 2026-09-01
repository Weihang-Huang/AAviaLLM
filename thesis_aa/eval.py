"""Evaluation protocol (Ch.4 of the thesis).

==========================================================================
  WHAT THIS MODULE DOES (for newcomers)
==========================================================================

After a method predicts an author for each test document, we need to score
how well it did. This module implements the metrics the thesis reports:

  * **macro-accuracy** — the unweighted average of per-author accuracy. This
    is the headline number in every results table. "Macro" means each
    author counts equally regardless of how many test documents they have,
    which matters when the test set is imbalanced.

  * **standard error (SE)** — the uncertainty on a mean accuracy estimate
    across repeated train/test splits. Reported in parentheses next to every
    accuracy in the thesis tables.

  * **top-N accuracy** — the fraction of documents whose true author is in
    the top-N predicted candidates (N = 1..5). Useful when there are many
    candidates (e.g. 50 in Blogs50): even if top-1 is wrong, top-5 might be
    right, which is still informative for real-world attribution.

  * **true-author rank** — the position of the true author in the ranked
    candidate list (1 = correctly attributed). Summarised as mean, std, and
    the 25/50/75/99th percentiles (Ch.5 Table 5).

  * **per-author accuracy** — accuracy broken down by author, so you can see
    which authors are easy/hard to attribute.

  * **benchmark results table** — the ``benchmark_results_df`` schema from
    ``CalculatePPL.ipynb``: one row per (feature, group) where group is either
    "GLOBAL" or one author tag, with columns F1 / precision / recall /
    accuracy. This is what the ALMs pipeline writes to disk.

All functions accept plain Python lists or numpy arrays, so they are easy to
unit-test without pandas. The benchmark aggregation helpers accept pandas
DataFrames matching the schema produced by ``alms/ppl.py``.

==========================================================================
  MACRO vs. MICRO ACCURACY (why "macro" matters)
==========================================================================

Suppose author A has 100 test docs and author B has 10. If a method gets 90
of A's right and 5 of B's right:

  * **micro accuracy** = (90 + 5) / 110 = 86.4% — A dominates the score.
  * **macro accuracy** = (0.90 + 0.50) / 2 = 70.0% — each author weighted
    equally, so B's poor performance is visible.

The thesis uses *macro* throughout because authorship-attribution benchmarks
vary in how many documents each author has, and we want to evaluate the
method's ability to distinguish authors, not its ability to do well on the
most populous author.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

_TAG_COL = "author_tag"


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def macro_accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    """Unweighted mean of per-class accuracy (Ch.4 Sec 4.3).

    "Macro" means every author counts equally regardless of how many test
    documents they have — a method that nails the populous author but fails
    the rare one gets punished, unlike micro accuracy.

    Example::

        y_true = ["a", "a", "b", "b", "c", "c"]
        y_pred = ["a", "a", "b", "b", "c", "a"]
        # per-class: a=2/2, b=2/2, c=1/2  ->  mean = (1+1+0.5)/3
        macro_accuracy(y_true, y_pred)   # 0.8333...
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    accs = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            continue
        accs.append(accuracy_score(y_true[mask], y_pred[mask]))
    return float(np.mean(accs)) if accs else 0.0


def standard_error(values: Sequence[float]) -> float:
    """Standard error of the mean of repeated-split accuracy estimates.

    Reported in parentheses next to every accuracy in the thesis tables,
    across repeated train/test splits.

    Example::

        standard_error([0.8, 0.8, 0.9])   # std(ddof=1) / sqrt(3)
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def top_n_accuracy(ranked_candidates: Sequence[Sequence], y_true: Sequence, n: int) -> float:
    """Macro-averaged top-N accuracy (Ch.5 Sec 5.4.1).

    ``ranked_candidates[i]`` is the ordered list of predicted candidates for
    document ``i`` (best first). Returns the unweighted mean, over true
    classes, of the proportion of that class's documents whose true author
    appears in the top ``n``. Useful with many candidates: if the true
    author makes your top-5 of 50, that is informative even when top-1
    is wrong.

    Example::

        y_true = ["a", "b"]
        ranked = [["a", "b"], ["b", "a"]]     # both correct at rank 1
        top_n_accuracy(ranked, y_true, 1)     # 1.0
        ranked = [["b", "a"], ["a", "b"]]     # both correct at rank 2
        top_n_accuracy(ranked, y_true, 1)     # 0.0
        top_n_accuracy(ranked, y_true, 2)     # 1.0
    """
    y_true = np.asarray(y_true)
    correct_flags = []
    for i, truth in enumerate(y_true):
        cand = list(ranked_candidates[i])[:n]
        correct_flags.append(int(truth in cand))
    correct_flags = np.asarray(correct_flags)

    accs = []
    for c in np.unique(y_true):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        accs.append(correct_flags[mask].mean())
    return float(np.mean(accs)) if accs else 0.0


def true_author_rank_stats(ranked_candidates: Sequence[Sequence], y_true: Sequence) -> dict:
    """Distribution of the true author's rank across documents (Ch.5 Table 5).

    The rank is 1-indexed: 1 means the true author was the top prediction.
    Absent truth degrades to ``len(candidates) + 1`` (worst case).

    Returns ``{"mean", "std", "Q25", "Q50", "Q75", "Q99"}`` — the percentiles
    summarise "how far down the list is the true author, typically?".

    Example::

        ranked = [["a", "b"], ["b", "a"]]
        true_author_rank_stats(ranked, ["a", "a"])
        # {"mean": 1.5, "std": 0.707..., "Q25": 1.25, "Q50": 1.5, ...}
    """
    y_true = np.asarray(y_true)
    ranks = []
    for i, truth in enumerate(y_true):
        cand = list(ranked_candidates[i])
        try:
            ranks.append(cand.index(truth) + 1)  # 1-indexed
        except ValueError:
            ranks.append(len(cand) + 1)  # worst-case if absent
    ranks = np.asarray(ranks, dtype=float)
    return {
        "mean": float(np.mean(ranks)),
        "std": float(np.std(ranks, ddof=1)) if len(ranks) > 1 else 0.0,
        "Q25": float(np.percentile(ranks, 25)),
        "Q50": float(np.percentile(ranks, 50)),
        "Q75": float(np.percentile(ranks, 75)),
        "Q99": float(np.percentile(ranks, 99)),
    }


def per_author_accuracy(y_true: Sequence, y_pred: Sequence) -> dict:
    """Per-class accuracy keyed by author tag.

    Example::

        per_author_accuracy(["a", "a", "b"], ["a", "b", "b"])
        # {"a": 0.5, "b": 1.0}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {}
    for c in np.unique(y_true):
        mask = y_true == c
        out[str(c)] = float(accuracy_score(y_true[mask], y_pred[mask]))
    return out


# ---------------------------------------------------------------------------
# Benchmark-result aggregation (mirrors CalculatePPL.ipynb output)
# ---------------------------------------------------------------------------

def build_benchmark_results_df(
    pred_df: pd.DataFrame,
    by_col: str = "by",
    true_col: str = "true_tag",
    pred_col: str = "pred_tag",
) -> pd.DataFrame:
    """Compute global + per-author benchmark rows for each ``by`` feature.

    Mirrors the ``pred_df_2_benchmark_results_df`` function in the original
    ``CalculatePPL.ipynb``: for each feature in ``by_col`` and for the global
    population plus each true author, compute macro F1 / precision / recall /
    accuracy.
    """
    rows = []
    # Sorted for deterministic output order (set iteration order varies
    # between runs; tests and notebook outputs should be reproducible).
    features = sorted(set(pred_df[by_col].tolist()))
    true_tags = sorted(set(pred_df[true_col].tolist()))

    for feature in features:
        sub = pred_df[pred_df[by_col] == feature]
        yt = sub[true_col].tolist()
        yp = sub[pred_col].tolist()

        rows.append(_metric_row(feature, "GLOBAL", yt, yp))
        for tag in true_tags:
            m = np.asarray(yt) == tag
            rows.append(_metric_row(feature, str(tag),
                                    np.asarray(yt)[m].tolist(),
                                    np.asarray(yp)[m].tolist()))

    return pd.DataFrame(rows)


def _metric_row(feature: str, tag: str, yt: list, yp: list) -> dict:
    if len(yt) == 0:
        return {"feature": feature, "true_tag": tag, "fscore": 0.0,
                "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    avg = "macro"
    zero = len(set(yt)) < 2  # can't compute macro metrics with one class
    return {
        "feature": feature,
        "true_tag": tag,
        "fscore": 0.0 if zero else f1_score(yt, yp, average=avg, zero_division=0),
        "precision": 0.0 if zero else precision_score(yt, yp, average=avg, zero_division=0),
        "recall": 0.0 if zero else recall_score(yt, yp, average=avg, zero_division=0),
        "accuracy": accuracy_score(yt, yp),
    }


# ---------------------------------------------------------------------------
# Summary table (cross-dataset, matching thesis tables)
# ---------------------------------------------------------------------------

def summarize_benchmark_dir(bench_dir: str, pattern: str = "*.csv") -> pd.DataFrame:
    """Read all benchmark result CSVs in ``bench_dir`` and return a summary.

    Each CSV is expected to have columns ``feature, true_tag, fscore,
    precision, recall, accuracy``. The returned frame averages over
    per-author rows (``true_tag != "GLOBAL"``) and keeps one row per
    (file, feature) — i.e. one **macro-accuracy per benchmark file**, the
    same aggregation the thesis tables report per dataset.

    Example::

        summarize_benchmark_dir(os.path.join(config.RESULTS_DIR,
                                             'benchmark_results_df_home'))
        # one row per (file, feature):
        #   file, feature, macro_accuracy, macro_fscore
    """
    import glob
    files = sorted(glob.glob(os.path.join(bench_dir, pattern)))
    out = []
    for f in files:
        df = pd.read_csv(f)
        per_author = df[df["true_tag"] != "GLOBAL"]
        for feature, grp in per_author.groupby("feature"):
            out.append({
                "file": os.path.basename(f),
                "feature": feature,
                "macro_accuracy": grp["accuracy"].mean(),
                "macro_fscore": grp["fscore"].mean(),
            })
    return pd.DataFrame(out)