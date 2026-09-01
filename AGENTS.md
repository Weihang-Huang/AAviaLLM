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
  data.py       # synthetic generator, benchmark loaders, subset downloader
  eval.py       # macro-acc, top-N, true-author rank, SE, benchmark writer
  alms/
    train.py    # further pretraining (ported from LMTrain-GPU.ipynb)
    ppl.py      # perplexity + CNLL (ported+fixed from CalculatePPL.ipynb)
  lerf/
    estimator.py  # LERF + 5 standard estimators + LMSE
    lerf_aa.py    # per-doc LERF + MFW + 8 classifiers
  tests/        # pytest smoke tests on synthetic data
notebooks/      # thin notebooks importing the modules
data/           # synthetic/ + benchmarks/ (downloaded subsets)
```

## Commands

- Run tests: `python -m pytest thesis_aa/tests/ -q`
- Lint/typecheck: none configured (no linter in repo).

## Notes for agents

- Defaults are CPU-safe and tiny (synthetic data, `gpt2`, 1-2 epochs) so
  pipelines run in seconds. For real runs, raise `epochs` to 100 (Table 2),
  swap to a benchmark via `data.load_benchmark`, and pick a larger GPT-2.
- `alms/ppl.py` fixes seven bugs from the original `CalculatePPL.ipynb`; see
  the module docstring for the list.
- Do not commit the downloaded benchmark archives or trained models.