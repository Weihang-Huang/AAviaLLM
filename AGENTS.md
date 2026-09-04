# AGENTS.md

## Project

Replication of three authorship-attribution methods from Weihang Huang's PhD
thesis "Authorship Attribution via Large Language Models" (University of
Birmingham), using the reference implementation at
https://github.com/Weihang-Huang/ALMs.

- **ALMs** (Ch.5) — further-pretrain one GPT-2 per candidate author; attribute
  by lowest perplexity; token-level CNLL for interpretability.
- **LERF** (Ch.6) — aggregate a *frozen* GPT-2's full next-token distributions
  to estimate vocabulary-wide relative frequencies; compare vs. MLE + 5
  standard estimators via LMSE.
- **LERF-AA** (Ch.7) — per-document LERF profiles + MFW selection + 8
  classifiers.

The repo contains ALMs notebooks only; LERF and LERF-AA were implemented from
the manuscript.

## Environment

- Python 3.13, torch 2.9 (xpu build, no CUDA), transformers 4.57, sklearn 1.9,
  spacy 3.8. CPU-only debugging by default.
- Install: `pip install -r requirements.txt`
- spaCy model (optional, for ALMs token tagging): `python -m spacy download en_core_web_sm`

## Layout

```
thesis_aa/
  config.py     # model sizes, paths, device
  data.py       # natural-corpus loader, synthetic generator, benchmark loaders
  eval.py       # macro-acc, top-N, true-author rank, SE, benchmark writer
  alms/
    train.py    # further pretraining (ported from LMTrain-GPU.ipynb)
    ppl.py      # perplexity + CNLL (ported+fixed from CalculatePPL.ipynb)
  lerf/
    estimator.py  # LERF + 5 standard estimators + LMSE (Eq. 6.9)
    lerf_aa.py    # per-doc LERF + observed-RF baseline + MFW + 8 classifiers
  tests/        # pytest suite (synthetic corpus; slow-marked natural-corpus tests)
scripts/
  generate_natural_corpus.py  # regenerates data/natural/ (needs local GPT-2)
notebooks/      # demo notebooks, all on the natural corpus
data/
  natural/      # shipped natural-English demo corpus (5 author personas, committed)
  synthetic/    # generated on the fly by generate_synthetic()
  benchmarks/   # downloaded subsets (gitignored)
```

## Commands

- Run tests: `python -m pytest thesis_aa/tests/ -q` (add `-m "not slow"` to
  skip model-heavy natural-corpus tests)
- Lint/typecheck: none configured (no linter in repo).

## Notes for agents

- Two demo corpora: `data/natural/` (shipped CSVs, natural-English prose by
  5 persona authors, 50 train / 20 test docs each — used by notebooks and
  the slow-marked directional tests) and `generate_synthetic()` (instant
  word salad — used by the fast test suite for pipeline mechanics). Do not
  describe how the natural corpus was produced in user-facing docs; just
  call it a small corpus of natural-English prose by five author personas
  in distinct topical domains.
- Defaults are CPU-safe and tiny (`gpt2`). The pytest suite uses 1-2 epochs
  on the instant synthetic corpus; the demo notebooks train ALMs at 15
  epochs on the natural corpus (~10 min/author on xpu). For real runs,
  raise `epochs` to 100 (Table 5.1), swap to a benchmark via
  `data.load_benchmark`, and pick a larger GPT-2.
- `alms/ppl.py` fixes seven bugs from the original `CalculatePPL.ipynb`; see
  the module docstring. README's bug tables (#8-#21) list the replication
  and manuscript-alignment fixes; each has a regression test.
- Do not commit the downloaded benchmark archives or trained models.