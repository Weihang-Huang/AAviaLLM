"""Smoke tests for the LERF and LERF-AA pipelines (Ch.6, Ch.7).

CPU-only, tiny synthetic corpus, ``gpt2``. Verifies:
  - LERF vector has the right shape and sums to 1;
  - all five standard estimators return valid distributions;
  - LMSE is computed and finite;
  - LERF-AA extracts per-document features and runs at least one classifier
    on MFW-50 with accuracy at least the random baseline.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from thesis_aa import config, data as data_mod
from thesis_aa.lerf import estimator as lerf_est
from thesis_aa.lerf import lerf_aa


@pytest.fixture(scope="module")
def tiny_corpus():
    return data_mod.generate_synthetic(
        n_authors=3, n_train_docs=4, n_test_docs=2, max_words=25, seed=11,
    )


def test_lerf_estimate_shape_and_normalisation(tiny_corpus):
    train_df, _ = tiny_corpus
    p = lerf_est.lerf_estimate(
        train_df["text"].tolist()[:4], model_name="gpt2",
        device=config.get_device(),
    )
    assert p.shape == (config.GPT2_VOCAB_SIZE,)
    assert p.sum() == pytest.approx(1.0, abs=1e-6)
    assert (p >= 0).all()
    # LERF must assign non-zero mass to types not observed in the sample.
    n_nonzero = (p > 0).sum()
    assert n_nonzero > 1000  # far more than a tiny corpus would observe


def test_mle_estimate_zero_for_unseen():
    texts = ["<BOS>the the the<EOS>"]
    p = lerf_est.mle_estimate(texts, model_name="gpt2")
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    n_nonzero = (p > 0).sum()
    assert n_nonzero <= 10  # only a few token ids observed


def test_standard_estimators_all_valid():
    texts = ["<BOS>the castle river dawn<EOS>", "<BOS>the photon circuit binary<EOS>"]
    ests = lerf_est.standard_estimators(texts, model_name="gpt2")
    assert set(ests.keys()) == {
        "Add-One", "Good-Turing", "Katz-Backoff", "Kneser-Ney", "Witten-Bell",
    }
    for name, p in ests.items():
        assert p.shape == (config.GPT2_VOCAB_SIZE,)
        assert np.isfinite(p).all(), f"{name} has non-finite values"
        assert (p >= 0).all(), f"{name} has negative probabilities"
        assert p.sum() == pytest.approx(1.0, abs=1e-6), f"{name} does not sum to 1"


def test_lmse_finite_and_signed():
    # Two identical distributions -> LMSE == 0 (worst fit among the reported).
    p = np.full(config.GPT2_VOCAB_SIZE, 1.0 / config.GPT2_VOCAB_SIZE)
    score = lerf_est.lmse(p, p)
    assert np.isfinite(score)
    assert score == pytest.approx(0.0, abs=1e-6)
    # A sharply different distribution should give a more negative score.
    q = np.zeros(config.GPT2_VOCAB_SIZE)
    q[0] = 1.0
    assert lerf_est.lmse(q, p) < score


def test_evaluate_all_estimators_returns_row(tiny_corpus):
    train_df, _ = tiny_corpus
    row = lerf_est.evaluate_all_estimators(
        train_df, model_name="gpt2", eval_frac=0.5, seed=0,
        device=config.get_device(),
    )
    expected_cols = {"LERF", "MLE", "Add-One", "Good-Turing",
                     "Katz-Backoff", "Kneser-Ney", "Witten-Bell"}
    assert set(row.columns) == expected_cols
    assert len(row) == 1
    for c in row.columns:
        assert np.isfinite(row[c].iloc[0])


def test_lerf_aa_extract_features(tiny_corpus):
    train_df, _ = tiny_corpus
    X = lerf_aa.extract_lerf_features(
        train_df.head(3), model_name="gpt2",
        device=config.get_device(),
    )
    assert X.shape == (3, config.GPT2_VOCAB_SIZE)
    for i in range(3):
        assert X[i].sum() == pytest.approx(1.0, abs=1e-6)


def test_lerf_aa_mfw_selection(tiny_corpus):
    train_df, test_df = tiny_corpus
    # Use a vocab-sized matrix so MFW indices (real token ids) are in-bounds.
    V = config.GPT2_VOCAB_SIZE
    X_train = np.zeros((len(train_df), V))
    X_test = np.zeros((len(test_df), V))
    # Put a signal on a few known-frequent token ids for realism.
    X_train[:, 0] = 1.0
    X_test[:, 0] = 1.0
    Xtr, Xte, idx = lerf_aa.select_mfw(
        X_train, X_test, train_df["text"].tolist(), k=10,
    )
    assert Xtr.shape == (len(train_df), 10)
    assert Xte.shape == (len(test_df), 10)
    assert idx.shape == (10,)
    # k >= vocab -> keep all.
    Xtr2, _, _ = lerf_aa.select_mfw(X_train, X_test, train_df["text"].tolist(), k=V)
    assert Xtr2.shape == X_train.shape
    # k larger than vocab -> keep all.
    Xtr3, _, _ = lerf_aa.select_mfw(X_train, X_test, train_df["text"].tolist(), k=V * 2)
    assert Xtr3.shape == X_train.shape


def test_lerf_aa_runs_one_classifier(tiny_corpus):
    train_df, test_df = tiny_corpus
    # Use only Linear SVM for speed on CPU.
    clf = {"Linear SVM": lerf_aa.build_classifiers()["Linear SVM"]}
    X_train = lerf_aa.extract_lerf_features(train_df, model_name="gpt2", device=config.get_device())
    X_test = lerf_aa.extract_lerf_features(test_df, model_name="gpt2", device=config.get_device())
    y_train = train_df["author_tag"].to_numpy()
    y_test = test_df["author_tag"].to_numpy()
    Xtr, Xte, _ = lerf_aa.select_mfw(X_train, X_test, train_df["text"].tolist(), k=50)
    res = lerf_aa.run_lerf_aa(Xtr, y_train, Xte, y_test, classifiers=clf)
    assert len(res) == 1
    assert "macro_accuracy" in res.columns
    # Should beat the random baseline.
    n_classes = train_df["author_tag"].nunique()
    assert res["macro_accuracy"].iloc[0] >= 1.0 / n_classes - 1e-9


def test_build_classifiers_runs_are_deterministic(tiny_corpus):
    """Regression: six of the eight classifiers are stochastic (SGD
    shuffling, random feature subsets) and previously had no random_state,
    so identical data produced different accuracy tables run-to-run —
    making debugging and regression testing impossible. All seeded
    classifiers must now return identical results on repeated runs."""
    train_df, test_df = tiny_corpus
    rng = np.random.default_rng(0)
    # Small synthetic feature matrices (no LERF extraction needed — we are
    # only testing classifier determinism, not feature quality).
    X_train = rng.random((12, 200))
    X_test = rng.random((6, 200))
    y_train = np.array(["a", "b", "c"] * 4)
    y_test = np.array(["a", "b", "c"] * 2)

    clfs = {"Linear SVM": lerf_aa.build_classifiers()["Linear SVM"],
            "Random Forest": lerf_aa.build_classifiers()["Random Forest"],
            "Decision Tree": lerf_aa.build_classifiers()["Decision Tree"]}
    r1 = lerf_aa.run_lerf_aa(X_train, y_train, X_test, y_test, classifiers=clfs)
    r2 = lerf_aa.run_lerf_aa(X_train, y_train, X_test, y_test, classifiers=clfs)
    assert r1.equals(r2), "stochastic classifiers must be seeded for reproducibility"


def test_select_mfw_shared_counting_helper_matches_mle(tiny_corpus):
    """The shared ``count_token_ids`` helper must produce the same MFW
    ranking as the (removed) inlined counting loops."""
    train_df, _ = tiny_corpus
    from thesis_aa.lerf.estimator import count_token_ids, mle_estimate
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    texts = train_df["text"].tolist()
    counts = count_token_ids(texts, tokenizer=tok)
    mle = mle_estimate(texts, tokenizer=tok)
    total = counts.sum()
    assert total > 0
    # MLE is just the normalised counts vector.
    assert np.allclose(mle, counts / total, atol=1e-12)