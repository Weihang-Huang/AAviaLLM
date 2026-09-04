"""Smoke tests for the ALMs pipeline (further pretraining + PPL + CNLL).

These run on CPU with a tiny synthetic corpus and ``gpt2`` so they finish in
seconds. They verify shapes, schemas, and that the fixed pipeline (the bugs
in CalculatePPL.ipynb) no longer manifest.
"""

import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from thesis_aa import config, data as data_mod, eval as eval_mod
from thesis_aa.alms import train as alms_train
from thesis_aa.alms import ppl as alms_ppl


# Common fixture: a tiny synthetic corpus + CPU device.
@pytest.fixture(scope="module")
def tiny_corpus():
    return data_mod.generate_synthetic(
        n_authors=3, n_train_docs=4, n_test_docs=2, max_words=25, seed=7,
    )


@pytest.fixture(scope="module")
def trained_models(tiny_corpus):
    train_df, _ = tiny_corpus
    with tempfile.TemporaryDirectory() as model_dir, tempfile.TemporaryDirectory() as log_dir:
        alms_train.train_all_authors(
            train_df,
            out_dir=model_dir,
            log_home=log_dir,
            epochs=1,
            gradient_accumulation_steps=1,
            batch_size=1,
            block_size=64,
            fp16=False,
        )
        yield model_dir


def test_train_produces_one_model_per_author(tiny_corpus, trained_models):
    train_df, _ = tiny_corpus
    expected = sorted(train_df["author_tag"].unique().tolist())
    found = sorted(d for d in os.listdir(trained_models)
                   if os.path.isdir(os.path.join(trained_models, d)))
    assert found == expected
    # Each model dir must contain a config.json.
    for tag in expected:
        assert os.path.isfile(os.path.join(trained_models, tag, "config.json"))


def test_train_is_resumable(tiny_corpus):
    train_df, _ = tiny_corpus
    with tempfile.TemporaryDirectory() as model_dir, tempfile.TemporaryDirectory() as log_dir:
        # Pre-write done.txt for one author.
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "done.txt"), "w", encoding="utf-8") as f:
            f.write("author00\n")
        alms_train.train_all_authors(
            train_df,
            out_dir=model_dir,
            log_home=log_dir,
            epochs=1,
            gradient_accumulation_steps=1,
            batch_size=1,
            block_size=64,
            fp16=False,
        )
        trained = os.listdir(model_dir)
        assert "author00" not in trained  # skipped
        assert len(trained) == train_df["author_tag"].nunique() - 1


def test_train_authorial_model_skip_check_fires_on_existing_model(tiny_corpus):
    """Regression: the skip-check used ``os.path.isdir`` on a *file* path
    (``models/<tag>/config.json``), which is always False — so an already
    trained author would silently retrain (hours wasted on real runs)
    whenever ``log/done.txt`` was missing. The check must be ``isfile``.
    """
    train_df, _ = tiny_corpus
    with tempfile.TemporaryDirectory() as model_dir:
        # Train one author directly.
        alms_train.train_authorial_model(
            "author00", train_df, out_dir=model_dir,
            epochs=1, gradient_accumulation_steps=1, batch_size=1,
            block_size=64, fp16=False,
        )
        marker = os.path.join(model_dir, "author00", "config.json")
        assert os.path.isfile(marker)
        # Second call with NO done.txt log: must detect the existing model
        # and skip (previously it retrained because isdir(file) is False).
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            path = alms_train.train_authorial_model(
                "author00", train_df, out_dir=model_dir,
                epochs=1, gradient_accumulation_steps=1, batch_size=1,
                block_size=64, fp16=False,
            )
        assert "already trained, skipping" in buf.getvalue()
        assert path == os.path.join(model_dir, "author00")


def test_score_all_pairs_writes_ce_logs_and_results(tiny_corpus, trained_models):
    _, test_df = tiny_corpus
    with tempfile.TemporaryDirectory() as work:
        ce_log_home = os.path.join(work, "ce_log")
        result_path = os.path.join(work, "ppl_result.csv")
        alms_ppl.score_all_pairs(
            None, test_df,
            model_dir=trained_models,
            ce_log_home=ce_log_home,
            result_path=result_path,
            use_tagger=False,
            limit_texts_per_author=2,
        )
        assert os.path.isfile(result_path)
        res = pd.read_csv(result_path)
        assert set(res.columns) == {"model_tag", "text_tag", "stride", "ppl"}
        assert len(res) == 3 * 3  # 3 model authors x 3 text authors
        ce_files = [f for f in os.listdir(ce_log_home) if f.endswith(".csv.7z")]
        assert len(ce_files) == 9


def test_aggregate_ppl_fixed_pipeline(tiny_corpus, trained_models):
    """Regression for bugs #1 and #2: ppl_df must be built from text_dfs."""
    _, test_df = tiny_corpus
    with tempfile.TemporaryDirectory() as work:
        ce_log_home = os.path.join(work, "ce_log")
        result_path = os.path.join(work, "ppl_result.csv")
        alms_ppl.score_all_pairs(
            None, test_df,
            model_dir=trained_models,
            ce_log_home=ce_log_home,
            result_path=result_path,
            use_tagger=False,
            limit_texts_per_author=2,
        )
        out_home = os.path.join(work, "ppl_dfs_buffer")
        paths = alms_ppl.aggregate_ppl(
            ce_log_home=ce_log_home, out_home=out_home,
        )
        assert len(paths) == 1
        ppl_df = pd.read_csv(paths[0])
        # Columns: true_tag, candidate_tag, text_num, global-ppl:(*)
        ppl_cols = [c for c in ppl_df.columns if c.startswith("global-ppl:")]
        assert len(ppl_cols) >= 2
        assert {"true_tag", "candidate_tag", "text_num"} <= set(ppl_df.columns)
        # Every PPL value is a positive finite number.
        for c in ppl_cols:
            assert np.isfinite(ppl_df[c]).all()
            assert (ppl_df[c] > 0).all()


def test_ce_log_losses_padding_invariant(tiny_corpus, trained_models):
    """The CE-log alignment convention: ``losses`` align to ``tokens[1:]``
    and, once padded for storage, end at exactly ``len(tokens)`` entries
    (one trailing 0.0 pad). Every downstream PPL average relies on this."""
    _, test_df = tiny_corpus
    with tempfile.TemporaryDirectory() as work:
        ce_log_home = os.path.join(work, "ce_log")
        result_path = os.path.join(work, "ppl_result.csv")
        alms_ppl.score_all_pairs(
            None, test_df,
            model_dir=trained_models,
            ce_log_home=ce_log_home,
            result_path=result_path,
            use_tagger=False,
            limit_texts_per_author=2,
        )
        import zipfile as _zip
        pair_fp = os.path.join(ce_log_home, "author00-author00.csv.7z")
        with _zip.ZipFile(pair_fp) as z:
            inner = [n for n in z.namelist() if n.endswith(".csv")][0]
            with z.open(inner) as f:
                corpus_df = pd.read_csv(f, names=alms_ppl.FEATURE_CATEGORIES,
                                        compression=None)
        import ast as _ast
        for _, row in corpus_df.iterrows():
            toks = _ast.literal_eval(str(row["tokens"]))
            losses = _ast.literal_eval(str(row["losses"]))
            # The CE-log convention: losses align to tokens[1:], possibly
            # already padded with one trailing 0.0.
            if len(losses) == len(toks) - 1:
                losses = losses + [0.0]
            assert len(losses) == len(toks), \
                "padded losses must have exactly tokens_count entries"


def test_predict_and_benchmark(tiny_corpus, trained_models):
    _, test_df = tiny_corpus
    with tempfile.TemporaryDirectory() as work:
        ce_log_home = os.path.join(work, "ce_log")
        result_path = os.path.join(work, "ppl_result.csv")
        alms_ppl.score_all_pairs(
            None, test_df,
            model_dir=trained_models,
            ce_log_home=ce_log_home,
            result_path=result_path,
            use_tagger=False,
            limit_texts_per_author=2,
        )
        out_home = os.path.join(work, "ppl_dfs_buffer")
        ppl_paths = alms_ppl.aggregate_ppl(
            ce_log_home=ce_log_home, out_home=out_home,
        )
        bench_paths = alms_ppl.predict_and_benchmark(
            ppl_paths,
            pred_home=os.path.join(work, "pred"),
            benchmark_home=os.path.join(work, "bench"),
        )
        bench = pd.read_csv(bench_paths[0])
        assert {"feature", "true_tag", "accuracy"} <= set(bench.columns)
        assert "GLOBAL" in bench["true_tag"].values


def test_train_authorial_model_single_document_author(tiny_corpus):
    """Regression: an author with exactly 1 training document previously
    crashed (datasets' train_test_split cannot split a single row into a
    non-empty train + eval set). Now the eval split is skipped and training
    proceeds on the one document."""
    train_df, _ = tiny_corpus
    one_doc = train_df[train_df["author_tag"] == "author00"].head(1).copy()
    rest = train_df[train_df["author_tag"] != "author00"].copy()
    df = pd.concat([one_doc, rest], ignore_index=True)
    with tempfile.TemporaryDirectory() as model_dir, tempfile.TemporaryDirectory() as log_dir:
        alms_train.train_all_authors(
            df,
            out_dir=model_dir,
            log_home=log_dir,
            epochs=1,
            gradient_accumulation_steps=1,
            batch_size=1,
            block_size=64,
            fp16=False,
        )
        # All authors trained, including the 1-doc author.
        trained = sorted(d for d in os.listdir(model_dir)
                         if os.path.isdir(os.path.join(model_dir, d)))
        assert "author00" in trained
        assert len(trained) == df["author_tag"].nunique()
        # The 1-doc author gets no eval.txt (no held-out eval split).
        assert not os.path.isfile(os.path.join(model_dir, "author00", "eval.txt"))
        # Multi-doc authors still get eval.txt.
        assert os.path.isfile(os.path.join(model_dir, "author01", "eval.txt"))


def test_compute_cnll_shape_and_sign():
    # 4 tokens, 3 authors. Author 0 is the true author (lowest NLL).
    nll = np.array([
        [0.1, 2.0, 3.0],
        [0.2, 1.5, 2.5],
        [0.3, 1.8, 2.8],
        [0.1, 2.1, 3.1],
    ])
    authors = ["a0", "a1", "a2"]
    cnll = alms_ppl.compute_cnll(nll, authors)
    assert cnll.shape == (4,)
    # CNLL for the best author should be negative (more predictable than avg of others).
    assert (cnll < 0).all()


def test_eval_metrics_basic():
    y_true = ["a", "a", "b", "b", "c", "c"]
    y_pred = ["a", "a", "b", "b", "c", "a"]
    assert eval_mod.macro_accuracy(y_true, y_pred) == pytest.approx((1 + 1 + 0.5) / 3)
    # top-1: true author at rank 1 for all 6 instances -> 1.0
    ranked = [["a", "b", "c"], ["a", "b", "c"], ["b", "c", "a"], ["b", "c", "a"],
              ["c", "a", "b"], ["c", "a", "b"]]
    assert eval_mod.top_n_accuracy(ranked, y_true, 1) == pytest.approx(1.0)
    assert eval_mod.top_n_accuracy(ranked, y_true, 2) == 1.0
    # Now build a ranking where the last c-instance has true author at rank 2.
    ranked2 = [["a", "b", "c"], ["a", "b", "c"], ["b", "c", "a"], ["b", "c", "a"],
               ["c", "a", "b"], ["a", "c", "b"]]
    assert eval_mod.top_n_accuracy(ranked2, y_true, 1) == pytest.approx((1 + 1 + 0.5) / 3)
    assert eval_mod.top_n_accuracy(ranked2, y_true, 2) == 1.0
    stats = eval_mod.true_author_rank_stats(ranked2, y_true)
    assert stats["mean"] == pytest.approx((1 + 1 + 1 + 1 + 1 + 2) / 6)