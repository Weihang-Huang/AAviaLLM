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