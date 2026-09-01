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
- 28 smoke tests pass on synthetic data with `gpt2` on CPU (`python -m pytest thesis_aa/tests/ -q`, ≈5-8 min).
- All three pipelines run end-to-end: ALMs (train → score → aggregate → benchmark), LERF (estimate → LMSE comparison), LERF-AA (extract features → MFW → classify).
- Output shapes and CSV schemas are correct; the 7 ALMs bug fixes (see below) are confirmed by regression tests.

**What is *not* yet verified:**
- The thesis accuracy tables (88.1%, 79.2%) have **not** been run on the full benchmarks. The smoke tests verify the *pipeline* runs, not these numbers.

---

## Results vs. thesis

| Method | Headline metric (thesis) | Thesis value | Reproduced? |
|---|---|---|---|
| ALMs | mean macro-accuracy across 4 datasets | **88.1%** | No — not yet run on full benchmarks |
| LERF | LMSE vs. 5 estimators across 7 corpora | LERF-XL best on 6/7 | No — not yet run |
| LERF-AA | mean macro-accuracy (full vocab, Linear SVM) | **79.2%** | No — not yet run |

The thesis values above are **reproduction targets**; the smoke tests verify
the pipeline runs end-to-end, not these numbers. See *Going to real data* for
how to attempt a full reproduction.

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
  tests/            # pytest smoke tests on synthetic data (28 tests)
notebooks/          # thin notebooks importing the modules
  ALMs_Train.ipynb
  ALMs_PPL.ipynb
  LERF_Estimate.ipynb
  LERF_AA.ipynb
data/               # synthetic/ (generated) + benchmarks/ (downloaded subsets)
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

### Run the tests (≈5 min, CPU)

```bash
python -m pytest thesis_aa/tests/ -q
# expect: 28 passed
```

### ALMs — train authorial models, then score and benchmark

```python
from thesis_aa import config, data as data_mod
from thesis_aa.alms import train as alms_train, ppl as alms_ppl

# tiny synthetic corpus (swap for data_mod.load_benchmark('Blogs50') on real data)
train_df, test_df = data_mod.generate_synthetic(n_authors=3, n_train_docs=6, n_test_docs=3)

# train one GPT-2 per author (epochs=2 for debug; thesis uses 100)
alms_train.train_all_authors(train_df, epochs=2, gradient_accumulation_steps=1,
                             batch_size=1, block_size=64, fp16=False)

# score every (model, author) pair, aggregate PPL, benchmark
result_path = alms_ppl.score_all_pairs(train_df, test_df, model_dir=config.MODEL_DIR)
ppl_paths = alms_ppl.aggregate_ppl()
bench_paths = alms_ppl.predict_and_benchmark(ppl_paths)
# What you'll see: [ALMs] author00: Perplexity: X.XX lines, then
# a benchmark_results_df_buffer-full.csv with macro-accuracy / F1 per author.
```

### LERF — estimate relative frequencies and compare estimators

```python
from thesis_aa import config
from thesis_aa.lerf import estimator as lerf_est

train_df, _ = data_mod.load_synthetic()
row = lerf_est.evaluate_all_estimators(train_df, model_name='gpt2',
                                       eval_frac=0.5, device=config.get_device())
print(row.T)
# What you'll see: a 7-column row (LERF, MLE, Add-One, Good-Turing,
# Katz-Backoff, Kneser-Ney, Witten-Bell), each an LMSE score (lower = better).
```

### LERF-AA — extract LERF features and classify

```python
from thesis_aa import config
from thesis_aa.lerf import lerf_aa

train_df, test_df = data_mod.load_synthetic()
results = lerf_aa.full_pipeline(train_df, test_df, model_name='gpt2',
                                mfw_sizes=[50, 100], device=config.get_device())
for k, df in results.items():
    print(f'--- MFW={k} ---')
    print(df[['classifier', 'macro_accuracy', 'top_1', 'top_5']])
# What you'll see: one CSV per MFW size under results/, with one row per
# classifier and columns for macro-accuracy and top-N accuracy.
```

---

## Notebooks

| Notebook | What it runs |
|---|---|
| `notebooks/ALMs_Train.ipynb` | Further-pretrains authorial GPT-2s on synthetic data. |
| `notebooks/ALMs_PPL.ipynb` | Scores pairs, aggregates PPL, writes benchmark CSVs. |
| `notebooks/LERF_Estimate.ipynb` | Evaluates LERF vs. the 5 standard estimators via LMSE. |
| `notebooks/LERF_AA.ipynb` | Full LERF-AA pipeline (features → MFW → 8 classifiers). |

Each is a thin wrapper that imports the corresponding module, so you can step
through the real pipeline in a notebook.

---

## Compute & time

| Path | Typical time |
|---|---|
| Synthetic + `gpt2` (tests / quick-start) | seconds |
| Synthetic + `gpt2-xl` | minutes |
| Blogs50 + `gpt2`, 100 epochs (thesis config) | ~hours per author × 50 authors |
| LERF-AA full vocab on a benchmark | minutes–hours |

The debug defaults (synthetic data, `gpt2`, 1–2 epochs) are chosen so every
pipeline runs in seconds on CPU. Real runs are dramatically heavier — plan
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
| 11 | `ppl.py` | `aggregate_ppl` re-decompressed and re-parsed every 7z CE log *per text-limit* | 17× redundant work on the thesis's 17-length sweep | Parse each log once, slice per limit (4×+ measured speedup, byte-identical output) |
| 12 | `data.py` | Module docstring referenced a nonexistent `_clean_text` helper | Newcomers chase a phantom function | Point at the real inline `.replace()` call sites |

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