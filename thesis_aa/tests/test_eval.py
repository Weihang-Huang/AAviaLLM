"""Tests for the evaluation metrics (Ch.4 protocol).

Pure-function tests: no models, no I/O — they run in milliseconds and pin
down the exact behaviour of every public metric, including the
deterministic-output fix in ``build_benchmark_results_df`` (row order must
not depend on set iteration).
"""

import pandas as pd
import pytest

from thesis_aa import eval as eval_mod


def test_macro_accuracy_weights_classes_equally():
    # Author 'a' has 100 docs (90 correct), author 'b' has 10 (5 correct):
    # micro would be 95/110 = 86.4%, macro is (0.9 + 0.5) / 2 = 70%.
    y_true = ["a"] * 100 + ["b"] * 10
    y_pred = ["a"] * 90 + ["x"] * 10 + ["b"] * 5 + ["y"] * 5
    assert eval_mod.macro_accuracy(y_true, y_pred) == pytest.approx(0.7)


def test_macro_accuracy_empty_and_single_class():
    assert eval_mod.macro_accuracy([], []) == 0.0
    assert eval_mod.macro_accuracy(["a", "a"], ["a", "b"]) == 0.5


def test_per_author_accuracy_dict():
    out = eval_mod.per_author_accuracy(["a", "a", "b"], ["a", "b", "b"])
    assert out == {"a": 0.5, "b": 1.0}


def test_standard_error():
    assert eval_mod.standard_error([0.8, 0.8, 0.9]) == pytest.approx(
        __import__("numpy").std([0.8, 0.8, 0.9], ddof=1) / (3 ** 0.5))
    assert eval_mod.standard_error([0.5]) == 0.0  # n=1 -> undefined -> 0
    assert eval_mod.standard_error([]) == 0.0


def test_true_author_rank_stats_absent_truth():
    # 'b' never appears in the second ranking -> worst-case rank len+1 = 3.
    ranked = [["a", "b"], ["a", "c"]]
    stats = eval_mod.true_author_rank_stats(ranked, ["a", "b"])
    assert stats["mean"] == pytest.approx((1 + 3) / 2)
    assert stats["Q50"] == pytest.approx(2.0)


def test_build_benchmark_results_df_is_deterministic():
    """Regression: features/true_tags were built from unordered sets, so
    CSV row order varied run-to-run. Output must now be stable."""
    pred_df = pd.DataFrame({
        "by": ["f1", "f1", "f2", "f2"],
        "true_tag": ["a", "b", "a", "b"],
        "pred_tag": ["a", "b", "b", "b"],
    })
    first = eval_mod.build_benchmark_results_df(pred_df)
    for _ in range(5):
        again = eval_mod.build_benchmark_results_df(pred_df)
        assert first.equals(again)
    # GLOBAL row exists for every feature, plus one row per (feature, author).
    assert set(first["true_tag"]) == {"GLOBAL", "a", "b"}
    # 2 features x (1 GLOBAL + 2 per-author rows) = 6 rows
    assert len(first) == 2 * 3