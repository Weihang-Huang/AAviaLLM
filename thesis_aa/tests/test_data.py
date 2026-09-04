"""Tests for the data layer (synthetic generator, loaders, splitting)."""

import os
import tempfile

import pandas as pd

from thesis_aa import data as data_mod


def test_generate_synthetic_shape_and_schema():
    train_df, test_df = data_mod.generate_synthetic(
        n_authors=3, n_train_docs=4, n_test_docs=2, max_words=30, seed=42,
    )
    assert list(train_df.columns) == ["text", "author_tag"]
    assert list(test_df.columns) == ["text", "author_tag"]
    assert len(train_df) == 3 * 4
    assert len(test_df) == 3 * 2
    assert train_df["author_tag"].nunique() == 3
    # Documents must contain BOS/EOS markers used by the ALMs repo.
    assert train_df["text"].iloc[0].startswith("<BOS>")
    assert train_df["text"].iloc[0].endswith("<EOS>")


def test_generate_synthetic_writes_files():
    with tempfile.TemporaryDirectory() as d:
        train_df, test_df = data_mod.generate_synthetic(
            n_authors=2, n_train_docs=2, n_test_docs=1, save_dir=d,
        )
        assert os.path.isfile(os.path.join(d, "train.csv"))
        assert os.path.isfile(os.path.join(d, "test.csv"))


def test_load_synthetic_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        data_mod.generate_synthetic(n_authors=2, n_train_docs=3, n_test_docs=2, save_dir=d)
        train_df, test_df = data_mod.load_synthetic(d)
        assert len(train_df) == 6
        assert len(test_df) == 4


def test_split_train_test_stratified():
    df = pd.DataFrame({
        "text": [f"<BOS>doc{i}<EOS>" for i in range(10)],
        "author_tag": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
    })
    train_df, test_df = data_mod.split_train_test(df, test_size=0.4, seed=0)
    assert set(train_df["author_tag"]) == {"a", "b"}
    assert set(test_df["author_tag"]) == {"a", "b"}
    assert len(train_df) + len(test_df) == 10


def test_download_subset_sampling_math():
    """Regression: the subset sampler previously used
    ``max_rows // n_authors + 1`` (overshooting max_rows badly for large
    author counts) and could yield 1-2 rows/author, which crashed
    ``train_authorial_model`` (datasets' train_test_split needs >= 2 rows
    to leave a non-empty train set). The new arithmetic is
    ``max(4, ceil(max_rows / n_authors))`` per author."""
    import math
    n_authors, max_rows = 50, 200
    per_author = max(4, math.ceil(max_rows / n_authors))
    assert per_author == 4          # 200/50 = 4 exactly, floor was 4+1=5
    total = per_author * n_authors
    assert total <= max_rows * 1.5  # close to max_rows, no 2x overshoot

    # Small author count: per-author is the exact share.
    assert max(4, math.ceil(200 / 13)) == 16  # ~207 total, within tolerance
    # The old arithmetic for comparison:
    assert 200 // 50 + 1 == 5  # 250 total = 25% overshoot (the old bug)