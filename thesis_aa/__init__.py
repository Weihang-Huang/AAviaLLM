"""Authorship Attribution via Large Language Models.

A replication package for Weihang Huang's PhD thesis
"Authorship Attribution via Large Language Models" (University of
Birmingham). Given a set of candidate authors and a questioned document,
decide *who wrote it* — using GPT-2 language models as the analysis engine.

 ==========================================================================
  NEW HERE? START WITH THIS MAP (2-minute orientation)
 ==========================================================================

The package implements the thesis's three methods. Read these in order to
follow the thesis's own storyline:

  1. **ALMs** (Ch.5) — ``thesis_aa.alms``
     *Further-pretrain* one GPT-2 per candidate author on that author's
     known writings, then attribute the questioned document to whichever
     author's model finds it most *predictable* (lowest perplexity).
     Submodules: ``train`` (build the models), ``ppl`` (score + attribute
     + interpret the decision token-by-token).

  2. **LERF** (Ch.6) — ``thesis_aa.lerf.estimator``
     With a single *frozen* GPT-2 (no training!), estimate how often each
     of GPT-2's 50,257 vocabulary types appears in a corpus — by summing
     the model's full next-token distributions over every context. Words
     the corpus never shows still get sensible estimates, because the
     model "knows" what English looks like.

  3. **LERF-AA** (Ch.7) — ``thesis_aa.lerf.lerf_aa``
     Use the LERF estimate as a *feature vector* (the "expected language"
     profile of a document) and feed it to standard classifiers. One
     shared frozen GPT-2 extracts features; eight classifiers compete.

Supporting modules (used by all three methods):

  - ``thesis_aa.config`` — the "control panel": paths, model sizes,
    thesis hyperparameters (``ALMS_TRAIN_CONFIG``), and device selection
    (``get_device()``: CUDA -> XPU -> CPU).
  - ``thesis_aa.data``  — synthetic corpus generator (fast debugging),
    benchmark loaders, and the downloader for the four real datasets.
  - ``thesis_aa.eval``  — metrics: macro-accuracy, top-N, true-author
    rank, standard error, and the benchmark-table builders.

 ==========================================================================
  YOUR FIRST FIVE MINUTES (copy-paste ready)
 ==========================================================================

.. code-block:: python

    from thesis_aa import config, data as data_mod
    from thesis_aa.alms import train as alms_train, ppl as alms_ppl

    # 1. A tiny 3-author practice corpus (generated locally in ms).
    train_df, test_df = data_mod.generate_synthetic(
        n_authors=3, n_train_docs=6, n_test_docs=3, seed=0)

    # 2. One GPT-2 per author (debug hyperparameters: seconds on CPU).
    alms_train.train_all_authors(train_df, epochs=2, block_size=64,
                                 batch_size=1, fp16=False)

    # 3. Score every (model, author) pair and attribute by lowest PPL.
    alms_ppl.score_all_pairs(train_df, test_df, model_dir=config.MODEL_DIR)
    ppl_paths = alms_ppl.aggregate_ppl()
    bench_paths = alms_ppl.predict_and_benchmark(ppl_paths)

See ``notebooks/`` for the fully annotated walk-throughs of all three
methods, and ``README.md`` for the reproduction targets (ALMs 88.1%,
LERF-AA 79.2% mean macro-accuracy on the four benchmarks).
"""

__version__ = "0.1.0"