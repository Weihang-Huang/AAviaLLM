# Authorship Attribution via Large Language Models

**Status: smoke tests passing on synthetic data; thesis numbers not yet reproduced.**

Replication of three authorship-attribution methods from Weihang Huang's PhD
thesis *"Authorship Attribution via Large Language Models"* (University of
Birmingham, 2026 manuscript). The reference implementation for the first
method lives at [`Weihang-Huang/ALMs`](https://github.com/Weihang-Huang/ALMs);
the other two were implemented from the manuscript.

In plain English: the thesis asks whether a GPT-2 can tell who wrote a mystery
text — by either *teaching one model per suspect author* and seeing whose
model finds the text most "predictable" (lowest perplexity), or by reading the
*probability distribution* a frozen GPT-2 assigns to every word and using that
distribution as a fingerprint for a standard classifier.

---

## The three methods at a glance

| Method | Thesis ch. | One-line idea | Module |
|---|---|---|---|
| **ALMs** | Ch.5 | Further-pretrain one GPT-2 per candidate author; attribute the questioned document to the author whose model gives it the **lowest perplexity**; decompose into token-level CNLL for interpretability. | `thesis_aa.alms` |
| **LERF** | Ch.6 | Aggregate a **frozen** GPT-2's full next-token distributions across all contexts in a corpus to estimate vocabulary-wide relative frequencies; compare vs. MLE + 5 standard estimators via LMSE. | `thesis_aa.lerf.estimator` |
| **LERF-AA** | Ch.7 | Treat each document as a sample corpus, extract its 50,257-dim **LERF profile**, apply MFW selection, and train **8 classifiers**. | `thesis_aa.lerf.lerf_aa` |

The thesis (Ch.8) frames ALMs and LERF-AA as two complementary views:
ALMs model *realised language* (the actual tokens in the questioned document),
while LERF-AA models *expected language* (the vocabulary-wide distribution
the LLM expects given the document's contexts).

---

## Reproducibility status

**What is verified:**
- 43 tests pass with `gpt2` (`python -m pytest thesis_aa/tests/ -q`, ≈11 min; use `-m "not slow"` to skip the model-heavy natural-corpus tests, ≈9 min).
- All three pipelines run end-to-end: ALMs (train → score → aggregate → benchmark), LERF (estimate → LMSE comparison), LERF-AA (extract features → MFW → classify).
- Output shapes and CSV schemas are correct; the 7 ALMs bug fixes, the 5 replication bug fixes, and the 10 manuscript-alignment fixes (see below) are confirmed by regression tests.
- The demo notebooks replicate the thesis's *directional* results on the shipped natural corpus: ALMs attributes the 3-author subset at ~90% macro-accuracy, LERF leads the LMSE table at the sparse evaluation share (Ch.6), and LERF features beat the observed-RF baseline at the smaller MFW sizes (Ch.7). Slow-marked regression tests pin these directions.

**What is *not* yet verified:**
- The thesis accuracy tables (88.1%, 79.2%) have **not** been run on the full benchmarks. The smoke tests verify the *pipeline* runs, not these numbers.

---

## Results vs. thesis

| Method | Headline metric (thesis) | Thesis value | Demo result (natural corpus) |
|---|---|---|---|
| ALMs | mean macro-accuracy across 4 datasets | **88.1%** | **90%** macro-acc on the 3-author notebook subset (epochs=15, gpt2) |
| LERF | LMSE vs. 5 estimators across 7 corpora | LERF-XL best on 6/7 | LERF best or 2nd at the 5% evaluation share (3/3 seeds beat MLE); at dense shares MLE is competitive — the thesis's own sparse-sample regime, observable at demo scale |
| LERF-AA | mean macro-accuracy (full vocab, Linear SVM) | **79.2%** | LERF features beat Observed-RF at MFW 50/100 (0.68/0.74 vs 0.51/0.54); full-vocab HistGB 0.93, RF 0.99. Linear SVM sits near chance at demo scale (5 personas × 50 train docs) — its 79.2% headline needs the real benchmarks |

The thesis values above are **reproduction targets**. The demo numbers are
directional replications at demo scale — the corpus is small, the model is
`gpt2` base, and ALMs trains 15 epochs instead of 100. See *Going to real
data* for how to attempt a full reproduction.

---

## Repository layout

```
thesis_aa/
  config.py        # model sizes, paths, device selection (the control panel)
  data.py           # synthetic generator, benchmark loaders, subset downloader
  eval.py           # macro-acc, top-N, true-author rank, SE, benchmark writer
  alms/
    train.py        # further pretraining (ported from LMTrain-GPU.ipynb)
    ppl.py          # perplexity + CNLL (ported + fixed from CalculatePPL.ipynb)
  lerf/
    estimator.py    # LERF + 5 standard estimators + LMSE
    lerf_aa.py      # per-doc LERF + MFW + 8 classifiers
  tests/            # pytest smoke tests (synthetic corpus; natural-corpus directional tests marked slow)
notebooks/          # thin notebooks importing the modules
  ALMs_Train.ipynb
  ALMs_PPL.ipynb
  LERF_Estimate.ipynb
  LERF_AA.ipynb
data/
  natural/          # shipped natural-English demo corpus (5 author personas)
  synthetic/        # generated on the fly by generate_synthetic()
  benchmarks/       # downloaded subsets (gitignored)
models/             # trained authorial GPT-2s (gitignored)
results/            # CE logs, PPL buffers, benchmark CSVs
log/                # ALMs resumability logs (done.txt / target.txt)
```

---

## Quick start

### Install

```bash
pip install -r requirements.txt
# optional: spaCy model for ALMs on-the-fly token tagging
python -m spacy download en_core_web_sm
```

### Run the tests (≈20 min, CPU)

```bash
python -m pytest thesis_aa/tests/ -q
# expect: 43 passed
# (add -m "not slow" to skip the model-heavy natural-corpus tests, ≈9 min)
```

### ALMs — train authorial models, then score and benchmark

```python
from thesis_aa import config, data as data_mod
from thesis_aa.alms import train as alms_train, ppl as alms_ppl

# natural demo corpus shipped with the repo (swap for data_mod.load_benchmark('Blogs50') on real data)
train_df, test_df = data_mod.load_natural()

# train one GPT-2 per author (epochs=15 for the demo; thesis uses 100)
alms_train.train_all_authors(train_df, epochs=15, gradient_accumulation_steps=1,
                             batch_size=1, block_size=64, fp16=False)

# score every (model, author) pair, aggregate PPL, benchmark
result_path = alms_ppl.score_all_pairs(train_df, test_df, model_dir=config.MODEL_DIR)
ppl_paths = alms_ppl.aggregate_ppl()
bench_paths = alms_ppl.predict_and_benchmark(ppl_paths)
# What you'll see: [ALMs] author00: Perplexity: X.XX lines, then
# a benchmark_results_df_buffer.csv with accuracy per author (~0.90
# macro-accuracy on the full 5-author demo corpus; ~0.90 on the
# 3-author notebook subset).
```

### LERF — estimate relative frequencies and compare estimators

```python
from thesis_aa import config
from thesis_aa.lerf import estimator as lerf_est

train_df, _ = data_mod.load_natural()
row = lerf_est.evaluate_all_estimators(train_df, model_name='gpt2',
                                       eval_frac=0.05, device=config.get_device())
print(row.T)
# What you'll see: a 7-column row (LERF, MLE, Add-One, Good-Turing,
# Katz-Backoff, Kneser-Ney, Witten-Bell), each an LMSE score (Ch.6 Eq. 6.9,
# lower = better) — with LERF at or near the top on this natural-English
# corpus at the sparse share, the demo-scale analogue of the thesis's
# Table 6.2 result. (At dense shares MLE is competitive — exactly the
# regime the thesis describes for its dense-sample corpora.)
```

### LERF-AA — extract LERF features and classify

```python
from thesis_aa import config
from thesis_aa.lerf import lerf_aa

train_df, test_df = data_mod.load_natural()
results = lerf_aa.full_pipeline(train_df, test_df, model_name='gpt2',
                                mfw_sizes=[50, 100], device=config.get_device())
for k, df in results.items():
    print(f'--- MFW={k} ---')
    print(df[['classifier', 'macro_accuracy', 'top_1', 'top_5']])

# The thesis's key Ch.7 comparison — LERF features vs the observed
# relative-frequency baseline (Tables 7.3 vs 7.4):
X_lerf = lerf_aa.extract_lerf_features(train_df, model_name='gpt2')
X_obs = lerf_aa.extract_observed_rf_features(train_df)
# ...feed both through select_mfw + run_lerf_aa with identical classifiers;
# LERF leads at the larger feature-set sizes.
```

---

## Notebooks

| Notebook | What it runs |
|---|---|
| `notebooks/ALMs_Train.ipynb` | Further-pretrains authorial GPT-2s on the natural demo corpus (3-author subset). |
| `notebooks/ALMs_PPL.ipynb` | Scores pairs, aggregates PPL, writes benchmark CSVs, CNLL token analysis. |
| `notebooks/LERF_Estimate.ipynb` | Evaluates LERF vs. the 5 standard estimators via LMSE on the natural corpus. |
| `notebooks/LERF_AA.ipynb` | LERF features vs the observed-RF baseline (Tables 7.3/7.4) + the 8-classifier pipeline. |

Each is a thin wrapper that imports the corresponding module, so you can step
through the real pipeline in a notebook. All four run on the natural demo
corpus shipped in `data/natural/` — no downloads needed after cloning.

---

## The demo corpora

The repository ships two small corpora so everything runs immediately
after cloning:

- **`data/natural/` (used by the notebooks)** — a small corpus of
  natural-English prose by five author personas, each writing in a
  distinct topical domain (castle life, ocean science, law, cookery,
  polar travel): 50 training + 20 test documents per author, standard
  `text,author_tag` CSV schema with `<BOS>`/`<EOS>` markers. Load with
  `data_mod.load_natural()`. The texts are ordinary English prose, so
  the demo notebooks replicate the thesis's *directional* results at
  demo scale — LERF leads the LMSE table at the sparse evaluation
  share (Ch.6), and LERF features beat the observed-RF baseline at the
  smaller MFW sizes (Ch.7), with ALMs attribution reaching ~90% macro
  accuracy on the 3-author notebook subset (Ch.5). Regenerate with
  `python scripts/generate_natural_corpus.py` (needs a local GPT-2;
  ~30 min on GPU-class hardware).
- **`generate_synthetic()` (used by the test suite)** — instant
  pseudo-English "word salad" with distinct per-author lexicons, generated
  on the fly (nothing stored). Fast enough for the pytest suite, which
  exercises pipeline mechanics (shapes, schemas, file plumbing) rather
  than estimation quality.

The real benchmarks (Blogs50, CCAT50, Guardian, IMDB62) remain the
reproduction targets — see *Going to real data*.

---

## Compute & time

Measured on the repo's development machine (xpu GPU, `gpt2`):

| Path | Typical time |
|---|---|
| Natural demo corpus + `gpt2`: LERF / LERF-AA pipelines | 1–2 min |
| ALMs training, natural corpus, 15 epochs | ~10 min per author |
| ALMs scoring + benchmark (3 authors) | ~3 min |
| Synthetic + `gpt2` (test suite) | seconds |
| `gpt2-xl` feature extraction (any corpus) | minutes |
| Blogs50 + `gpt2`, 100 epochs (thesis config) | ~hours per author × 50 authors |
| LERF-AA full vocab on a benchmark | minutes–hours |

The pytest defaults (synthetic data, `gpt2`, 1–2 epochs) keep the fast
suite in seconds on CPU. Real runs are dramatically heavier — plan
accordingly, and use a GPU.

---

## Bugs fixed from the original ALMs notebook

The reference `CalculatePPL.ipynb` from `Weihang-Huang/ALMs` contained several
bugs that broke or corrupted the aggregation pipeline. All are fixed here and
documented inline in `thesis_aa/alms/ppl.py`.

| # | File | Bug | Impact | Fix |
|---|---|---|---|---|
| 1 | `ppl.py` | `predict_and_benchmark` grouped by `text_num` alone, conflating documents from different authors that share the same `text_num` | Wrong predictions; corrupted accuracy & doc counts | Group by `(true_tag, text_num)` |
| 2 | `ppl.py` | Per-token losses double-counted context tokens in overlapping sliding windows (`shift_labels` from `input_ids`, not the masked `target_ids`) | Inflated/skewed perplexity on long texts | Derive `shift_labels` from `target_ids`; drop masked zeros |
| 3 | `train.py` | `fp16` could never be True because it was gated on `config.ALMS_TRAIN_CONFIG["fp16"]` (=False) | Training slower than necessary on GPU | Auto-detect CUDA; ignore the static config flag |
| 4 | `data.py` | `_make_doc` crashed when `max_words < 20` (`randint(20, max_words)`) | Test crash on tiny debug corpora | Use `min(20, max_words)` as the lower bound |
| 5 | `estimator.py` | `_good_turing` claimed to assign mass to unseen types but never did | Unseen types got 0 probability, contradicting the method | Implement the standard `N1/N` mass split across unseen types |
| 6 | `train.py` | Referenced `config.TAG_COL` which doesn't exist (it lives in `data.py`) | Fragile default; misleading to readers | Use a plain `"author_tag"` default |
| 7 | multiple | Dead imports (`io`, `zipfile`, `math`, `os`, `F`, `Iterable`) and unused parameters (`batch_size`, `ranked_candidates`) | Clutter; confusing for newcomers | Removed |

In short: the original scoring notebook **did not run end-to-end as
published**; this one does. The full audit trail is in the `alms/ppl.py`
module docstring.

### Bugs found in this replication itself (2026 audit)

A systematic debug pass over the replication found and fixed these
additional issues — each has a regression test:

| # | File | Bug | Impact | Fix |
|---|---|---|---|---|
| 8 | `train.py` | The "already trained?" skip-check used `os.path.isdir()` on a *file* path (`models/<tag>/config.json`) — always False | An already-trained author silently retrained (hours wasted on real runs) whenever `log/done.txt` was missing | Use `os.path.isfile()` |
| 9 | `lerf_aa.py` | Six of the eight classifiers were stochastic (SGD shuffle, random feature subsets, AdaBoost sampling) with no `random_state` | Identical data produced different accuracy tables run-to-run; regression testing impossible | Pin `random_state=0` on every stochastic classifier |
| 10 | `eval.py` | `build_benchmark_results_df` iterated `set(...)` — nondeterministic row order | Benchmark CSVs not byte-reproducible across runs | Sort features/true_tags |
| 11 | `data.py` | Module docstring referenced a nonexistent `_clean_text` helper | Newcomers chase a phantom function | Point at the real inline `.replace()` call sites |

### Manuscript-alignment fixes (2026 audit, round 2)

A second audit pass cross-checked every module against the thesis manuscript
(`manuscript.docx`). Each fix below cites the manuscript section that
specifies the corrected behaviour, and each has a regression test:

| # | File | Bug | Manuscript basis | Fix |
|---|---|---|---|---|
| 13 | `estimator.py` | `lmse` computed `−mean((log p̂ − log p)²)` with an `eps` floor — a different metric from the thesis, with *inverted* ranking (the code's own convention was higher=better while docs claimed lower=better); zero-mass reference types dominated every score via `log(1e-12)` | Eq. 6.9: "the mean of the squared differences between estimated and ground-truth frequencies … then take the base-2 logarithm"; Table 6.2: "Lower (more negative) values indicate superior performance"; §6.5.2: target vocabulary = types observed in the reference corpus | `log2(mean((p̂ − p)²))` over reference-observed types, no eps, no negation; perfect fit = −inf |
| 14 | `estimator.py` | `split_corpus`/`evaluate_all_estimators` defaulted `eval_frac=0.2` | §6.5.2: "We have therefore chosen to draw an evaluation corpus that consists of 50% of the word tokens found in each reference corpus" | Default `eval_frac=0.5` |
| 15 | `estimator.py` | `_katz_backoff` returned a non-distribution (measured sum 1.6) when the freq-of-freq curve was non-monotone (GT discount factor > 1) | Eq. 6.4 defines "a normalization factor that distributes the freed probability over the backed-off distribution" | Clamp freed mass at ≥ 0 (a discount never adds mass); renormalise to sum 1 |
| 16 | `estimator.py` | `lerf_estimate` used `stride = 1024 = n_positions`: non-overlapping chunks — every token after a chunk boundary was predicted with *zero* prior context, and boundary positions were skipped | §6.4.3: each token's context is "the sequence of words that precedes it"; Ch.3: the sliding-window procedure "should therefore be reported" | Overlapping windows (window = `n_positions`, default stride = half); each position counted exactly once with a context floor of `n_positions − stride` tokens; windowing documented |
| 17 | `lerf_aa.py` | Binary-class `decision_function` scores assigned to the wrong class columns (score of `classes_[1]` went to `classes_[0]`'s column) | §7.3.3: "the candidate with the highest value is returned as the predicted author" | Expand to `[-S, +S]` so the positive class's score lands on its own column; predicted class always ranks first |
| 18 | `lerf_aa.py` | AdaBoost stumps had no `max_features` (the code even *documented* sqrt); HistGB used `max_features=1.0` | §7.3.3 verbatim: stumps use "the square root of the available features considered at its split"; HistGB uses "10% of the available features during fitting" | Stump `max_features="sqrt"`; HistGB `max_features=0.1` |
| 20 | `data.py` + `train.py` | `download_benchmark_subset` used `max_rows // n_authors + 1` (overshooting `max_rows` by up to 25%) and could yield 1–2 rows/author, which crashed training (`datasets.train_test_split` cannot split a single row) | Robustness (no manuscript conflict); README documents subset sanity-checks | `max(4, ceil(max_rows / n_authors))` per author; a 1-doc author now trains without a held-out eval split (clear message, no eval.txt) instead of crashing |
| 21 | `estimator.py` | `split_corpus` returned *disjoint* eval/reference halves | §6.5.2: "the reference corpus always contains the evaluation corpus as well as additional texts that were randomly sampled … when the evaluation corpus was drawn" | Reference = the full input corpus; evaluation = a random text subset of it (containment) |

Two earlier rows were removed along with the machinery they described: the
original notebook's **text-length sweep** (`test_text_limits`, producing
`buffer-10`/`buffer-20`/... artifacts). It is absent from the manuscript
(§5.4.1 studies accuracy on full-length question texts only), so the
sweep — and the bugs that only existed in its truncation arithmetic —
were deleted wholesale rather than documented.

A third row (#19, the F1/precision/recall zero-gate in `_metric_row`) met
the same fate in a later alignment pass: the manuscript reports
**accuracy only** in every results table, so the F1/precision/recall
machinery inherited from `CalculatePPL.ipynb` (and `summarize_benchmark_dir`'s
`macro_fscore` aggregate) was deleted outright rather than fixed, leaving
accuracy-only benchmark CSVs and summaries.

---

## Going to real data

### 1. Get the benchmarks

The four benchmarks used in the thesis are available as `.csv.7z` archives in
the [original repo's `data/` directory](https://github.com/Weihang-Huang/ALMs/tree/main/data):
**Blogs50**, **CCAT50**, **Guardian**, **IMDB62**.

Download a small stratified subset for a sanity check:

```python
from thesis_aa import data as data_mod
train_df, test_df = data_mod.download_benchmark_subset('Blogs50', max_rows=200)
```

Or place the full archives under `data/benchmarks/<Name>/` as `train.csv` /
`test.csv` and load them directly:

```python
train_df, test_df = data_mod.load_benchmark('Blogs50')
```

### 2. Pick a larger GPT-2

```python
from thesis_aa import config
config.MODEL_SIZES['xl']   # ('gpt2-xl', '1.5B') — the thesis default for LERF-AA
```

Pass `model_name='gpt2-xl'` to `lerf_aa.full_pipeline` /
`lerf_est.evaluate_all_estimators`, and `base_model='gpt2'` (or larger) to
`alms_train.train_all_authors`.

### 3. Set the thesis hyperparameters

```python
from thesis_aa import config
config.ALMS_TRAIN_CONFIG
# {'epochs': 100, 'learning_rate': 2e-5, 'weight_decay': 0.01,
#  'gradient_accumulation_steps': 64, 'block_size': 128, 'fp16': False}
```

For ALMs: `epochs=100`, `gradient_accumulation_steps=64`, `block_size=128`
(Ch.5 Table 2). For LERF-AA: full 50,257-type vocabulary + Linear SVM
(Ch.7 Sec 7.4.1).

### 4. GPU note

`fp16` auto-enables only when **CUDA** is available. The `xpu` build of torch
used in this repo's development does *not* support `Trainer`'s fp16 path —
use CUDA or run in fp32.

### What to expect

ALMs on Blogs50 with 100 epochs per author is the heaviest path: roughly
hours per author × 50 authors on a single A100 (the thesis used one A100).
LERF-AA is lighter — feature extraction dominates and runs in
minutes-to-hours per benchmark depending on model size and document count.

---

## Evaluation metrics

All metrics live in `thesis_aa/eval.py` and follow the Ch.4 protocol:

- **macro-accuracy** — the unweighted mean of per-author accuracy. The headline number in every thesis table. "Macro" means each author counts equally regardless of how many test documents they have.
- **top-N accuracy** — fraction of documents whose true author is in the top-N predicted candidates (N = 1..5). Useful with many candidates.
- **true-author rank** — position of the true author in the ranked list (1 = correct); reported as mean, std, and 25/50/75/99th percentiles.
- **standard error** — uncertainty on a mean accuracy across repeated splits.
- **per-author accuracy** — accuracy broken down by author.

**Why macro, not micro?** Suppose author A has 100 test docs and author B has
10; a method gets 90 of A's and 5 of B's right. Micro accuracy = 95/110 = 86%,
so A dominates the score. Macro accuracy = (0.90 + 0.50)/2 = 70%, so B's poor
performance is visible. The thesis uses macro throughout.

---

## Troubleshooting

- **ALMs token annotations are empty.** You need a spaCy model:
  `python -m spacy download en_core_web_sm`. Without it, ALMs falls back to
  empty annotation columns (the pipeline still runs; you just lose POS/lemma/etc.).
- **`download_benchmark_subset` fails.** The benchmark archives are `.7z`;
  you need `py7zr` (`pip install py7zr`) or a `7z` executable on your PATH.
- **`fp16` errors on an Intel GPU.** The `xpu` torch build doesn't support
  `Trainer`'s fp16 path. Pass `fp16=False`, or use a CUDA build of torch.
- **Long ALMs run interrupted.** Safe to restart — completed authors are
  logged in `log/done.txt` and skipped automatically.

---

## What's not included / limitations

- **LERF and LERF-AA had no published code.** They were implemented from the
  manuscript; the only reference code in `Weihang-Huang/ALMs` is the ALMs notebooks.
- **English benchmarks only**, and only the GPT-2 model family. The thesis
  itself is closed-set attribution; open-set / verification are out of scope.
- **spaCy tagger is optional.** The ALMs on-the-fly token tagging falls back
  to empty annotations if no spaCy model is installed.
- **License:** to be determined. The original `Weihang-Huang/ALMs` repo is
  Apache-2.0; this repo's license has not yet been chosen.

---

## Citation & attribution

Based on Weihang Huang's PhD thesis *“Authorship Attribution via Large Language
Models”* (University of Birmingham, 2026) and the reference implementation at
[github.com/Weihang-Huang/ALMs](https://github.com/Weihang-Huang/ALMs).

Placeholder BibTeX (fill in the exact fields once the thesis is deposited):

```bibtex
@phdthesis{huang2026authorship,
  title  = {Authorship Attribution via Large Language Models},
  author = {Huang, Weihang},
  school = {University of Birmingham},
  year   = {2026},
}
```

---

## See also

`AGENTS.md` has the condensed agent-facing summary of the same information.