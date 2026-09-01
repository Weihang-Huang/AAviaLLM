"""Data layer: benchmark loaders, synthetic generator, and real-subset sampler.

==========================================================================
  WHAT THIS MODULE DOES (for newcomers)
==========================================================================

Every experiment in the thesis starts with a corpus of texts labelled by
author. This module provides three ways to get such a corpus:

  1. **``generate_synthetic``** — create a tiny fake corpus on the fly. Each
     author has a distinct "lexicon" of content words (castles, circuits,
     gardens, harbours, ledgers) so attribution is actually learnable. This
     is what the tests and CPU debugging use: it runs in milliseconds and
     needs no network. The schema matches the real benchmarks so you can
     swap it out for real data without changing any downstream code.

  2. **``load_benchmark``** / **``load_synthetic``** — load a previously
     saved corpus (real or synthetic) from disk. Real benchmarks come as
     ``train.csv`` / ``test.csv`` pairs under ``data/benchmarks/<name>/``.

  3. **``download_benchmark_subset``** — fetch a benchmark CSV archive from
     the ALMs GitHub repo, extract it, and sample a small stratified subset.
     Useful for sanity-checking on real data without downloading the full
     corpus. Requires ``py7zr`` (or a ``7z`` executable) for extraction.

All loaders return ``(train_df, test_df)`` — two pandas DataFrames with
columns ``["text", "author_tag"]``. The ``text`` column contains documents
wrapped in ``<BOS>`` / ``<EOS>`` markers, matching the schema used in the
ALMs repo's benchmark CSVs.

==========================================================================
  SYNTHETIC GENERATOR — how it works
==========================================================================

There are five author "personas", each with a 15-word content lexicon:

    author00 -> castle, river, honour, blade, kingdom, ...
    author01 -> circuit, photon, quantum, array, neuron, ...
    author02 -> garden, blossom, meadow, willow, petal, ...
    author03 -> harbour, tide, vessel, compass, mariner, ...
    author04 -> ledger, audit, equity, dividend, portfolio, ...

Each document is a random sequence of ~20-60 words, drawn with probabilities
that bias ~50% toward the author's lexicon and ~30% toward a shared set of
common function words (the, of, and, ...). This gives each author a
recognisable "signature" that the LLM-based methods can learn, while keeping
the texts short enough for CPU debugging. Pass ``max_words`` smaller than 20
if you want very short documents; the generator handles that gracefully.

==========================================================================
  CSV SCHEMA (must match the ALMs repo)
==========================================================================

    text,author_tag
    <BOS>...document body...<EOS>,author00
    <BOS>...document body...<EOS>,author00
    ...

The ``<BOS>`` / ``<EOS>`` markers are kept in the CSV files so the on-disk
format matches the reference repo exactly; each pipeline strips them with
inline ``text.replace("<BOS>", "").replace("<EOS>", "")`` calls before
tokenisation (see ``alms/ppl.py:compute_ce_per_text`` and
``lerf/estimator.py:lerf_estimate``).
"""

from __future__ import annotations

import os
import random
import urllib.request
from typing import Tuple

import pandas as pd

from . import config

TEXT_COL = "text"
TAG_COL = "author_tag"


# ---------------------------------------------------------------------------
# Synthetic corpus generator
# ---------------------------------------------------------------------------

_AUTHOR_LEXICONS = {
    0: ["castle", "river", "honour", "blade", "kingdom", "oath", "gallant",
        "banner", "knight", "throne", "siege", "valor", "crown", "forge", "dawn"],
    1: ["circuit", "photon", "quantum", "array", "neuron", "binary", "kernel",
        "sensor", "voltage", "protocol", "matrix", "cache", "thread", "laser", "robot"],
    2: ["garden", "blossom", "meadow", "willow", "petal", "breeze", "sparrow",
        "harvest", "orchard", "morning", "sunlight", "dew", "meadow", "fern", "lilac"],
    3: ["harbour", "tide", "vessel", "compass", "mariner", "anchor", "gull",
        "rigging", "voyage", "current", "helm", "lighthouse", "shoal", "mast", "wake"],
    4: ["ledger", "audit", "equity", "dividend", "portfolio", "bond", "shareholder",
        "arbitrage", "forecast", "revenue", "asset", "volatility", "treasury", "merger", "cap"],
}

_COMMON_WORDS = ["the", "of", "and", "to", "in", "a", "was", "he", "she", "it",
                 "that", "on", "for", "with", "as", "by", "at", "this", "from",
                 "but", "not", "they", "his", "her", "we", "all", "were", "one"]


def _make_doc(author_idx: int, max_words: int, rng: random.Random) -> str:
    """Generate one pseudo-English document biased toward an author's lexicon.

    ``max_words`` is the upper bound on the document length. The lower bound is
    ``min(20, max_words)`` so very small ``max_words`` values (used in CPU
    debug tests) do not cause ``randint`` to raise ``ValueError``.
    """
    lex = _AUTHOR_LEXICONS[author_idx % len(_AUTHOR_LEXICONS)]
    lo = min(20, max_words)
    n = rng.randint(lo, max_words) if max_words > 1 else 1
    words: list[str] = []
    for _ in range(n):
        r = rng.random()
        if r < 0.35:
            words.append(rng.choice(lex))
        elif r < 0.80:
            words.append(rng.choice(_COMMON_WORDS))
        else:
            words.append(rng.choice(lex))
    body = " ".join(words)
    return f"<BOS>{body}<EOS>"


def generate_synthetic(
    n_authors: int = 5,
    n_train_docs: int = 10,
    n_test_docs: int = 4,
    max_words: int = 60,
    seed: int = 0,
    save_dir: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a tiny synthetic authorship corpus.

    Produces ``n_authors`` authors each with ``n_train_docs`` / ``n_test_docs``
    documents. Each author draws from a distinct content-word lexicon so that
    attribution is learnable. Returns ``(train_df, test_df)``.
    """
    rng = random.Random(seed)
    rows_tr, rows_te = [], []
    for a in range(n_authors):
        tag = f"author{a:02d}"
        for _ in range(n_train_docs):
            rows_tr.append({TEXT_COL: _make_doc(a, max_words, rng), TAG_COL: tag})
        for _ in range(n_test_docs):
            rows_te.append({TEXT_COL: _make_doc(a, max_words, rng), TAG_COL: tag})
    train_df = pd.DataFrame(rows_tr)
    test_df = pd.DataFrame(rows_te)

    if save_dir is None:
        save_dir = config.SYNTHETIC_DIR
    os.makedirs(save_dir, exist_ok=True)
    train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False, encoding="utf-8")
    return train_df, test_df


# ---------------------------------------------------------------------------
# Benchmark loaders
# ---------------------------------------------------------------------------

def load_csv_split(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load ``train.csv`` and ``test.csv`` from ``data_dir``."""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    if not os.path.isfile(train_path) or not os.path.isfile(test_path):
        raise FileNotFoundError(
            f"Expected train.csv and test.csv in {data_dir}."
        )
    train_df = pd.read_csv(train_path, encoding="utf-8")
    test_df = pd.read_csv(test_path, encoding="utf-8")
    for df in (train_df, test_df):
        if TEXT_COL not in df.columns or TAG_COL not in df.columns:
            raise ValueError(
                f"CSV must contain '{TEXT_COL}' and '{TAG_COL}' columns; "
                f"found {list(df.columns)}"
            )
    return train_df, test_df


def load_benchmark(name: str, root: str | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load a benchmark corpus from ``root/<name>/``."""
    if root is None:
        root = config.BENCHMARK_DIR
    return load_csv_split(os.path.join(root, name))


def load_synthetic(root: str | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the most recently generated synthetic corpus."""
    if root is None:
        root = config.SYNTHETIC_DIR
    return load_csv_split(root)


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a single dataframe into train/test by author-tag, stratified.

    Mirrors the per-author 80/20 split used in ``LMTrain-GPU.ipynb``.
    """
    rng = random.Random(seed)
    train_parts, test_parts = [], []
    for tag, group in df.groupby(TAG_COL):
        idx = list(group.index)
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * (1 - test_size)))
        train_parts.append(df.loc[idx[:cut]])
        test_parts.append(df.loc[idx[cut:]])
    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Real-subset downloader (for sanity checks without the full corpus)
# ---------------------------------------------------------------------------

def download_benchmark_subset(
    name: str,
    max_rows: int = 50,
    out_dir: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download a benchmark CSV archive from the ALMs repo and sample rows.

    The repo stores single CSVs (e.g. ``Blogs50.csv.7z``), not pre-split.
    We download, extract, split by author, and return small train/test frames.
    """
    if name not in config.BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {name}")
    if out_dir is None:
        out_dir = os.path.join(config.BENCHMARK_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    url = f"{config.REPO_DATA_URL}/{name}.csv.7z"
    archive_path = os.path.join(out_dir, f"{name}.csv.7z")

    if not os.path.isfile(archive_path):
        urllib.request.urlretrieve(url, archive_path)

    # Extract the inner CSV from the 7z archive.
    csv_path = os.path.join(out_dir, f"{name}.csv")
    if not os.path.isfile(csv_path):
        try:
            import py7zr
            with py7zr.SevenZipFile(archive_path, mode="r") as z:
                z.extractall(path=out_dir)
        except ImportError:
            # Fallback: try the `7z` CLI if available.
            import shutil
            import subprocess
            exe = shutil.which("7z") or shutil.which("7za")
            if exe is None:
                raise RuntimeError(
                    "Need py7zr or a 7z executable to extract the archive."
                )
            subprocess.run([exe, "x", "-y", f"-o{out_dir}", archive_path], check=True)

    full = pd.read_csv(csv_path, encoding="utf-8")
    if TEXT_COL not in full.columns or TAG_COL not in full.columns:
        raise ValueError(
            f"Downloaded {name} missing required columns: {list(full.columns)}"
        )

    # Sample a small stratified subset.
    sampled = (
        full.groupby(TAG_COL, group_keys=False)
        .apply(lambda g: g.head(max_rows // full[TAG_COL].nunique() + 1))
        .reset_index(drop=True)
    )
    train_df, test_df = split_train_test(sampled, test_size=0.2, seed=0)
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False, encoding="utf-8")
    return train_df, test_df