"""Perplexity attribution and token-level analysis for ALMs (Ch.5 Sec 5.3.2-5.3.3).

==========================================================================
  WHAT THIS MODULE DOES (big picture for newcomers)
==========================================================================

This module is the *scoring* half of the ALMs method. After
``alms/train.py`` has fine-tuned one GPT-2 per candidate author, this module
takes those saved models and answers the question:

    "Given a questioned document, which candidate author's model finds it
     most predictable (lowest perplexity)?"

The answer is the predicted author. The module also breaks the document-level
score down to the *token* level, so you can see exactly which words in the
questioned document contributed most to the attribution. This is the
"interpretability" advantage the thesis claims for ALMs over black-box
classifiers.

There are three stages, each a public function:

  1. ``score_all_pairs``  — score every (author-model, test-author) pair and
                             write per-token cross-entropy logs + a PPL table.
  2. ``aggregate_ppl``    — turn the raw per-token logs into one per-text PPL
                             value.
  3. ``predict_and_benchmark`` — attribute each test text to the argmin-PPL
                             author and compute accuracy.

A fourth function, ``compute_cnll``, implements the token-level Comparative
Negative Log-Likelihood from Ch.5 Sec 5.3.3 (used for interpretability).

==========================================================================
  KEY CONCEPTS (quick glossary)
==========================================================================

  **Perplexity (PPL)**  — exp(mean NLL). A measure of how "surprising" a text
     is to a language model. Lower = more predictable = more likely written by
     that model's author. Range: 1 (perfect) to +inf. See Ch.5 Eq. 1.

  **NLL (negative log-likelihood)** — -log p(x_i | x_{<i}). The per-token
     "surprisal" of the actual next token under the model. High NLL = the
     model did not see that token coming. See Ch.5 Eq. 2.

  **CNLL (comparative NLL)** — NLL under one candidate minus the *mean* NLL
     under all other candidates. Negative => this token favours that
     candidate; positive => it disfavours her. See Ch.5 Eq. 3-4.

  **Sliding-window scoring** — GPT-2 has a fixed context length (1024 tokens
     for gpt2-base). Texts longer than that are scored in overlapping windows
     of size ``max_length`` advanced by ``stride``. Each window masks the
     context it shares with the previous window so tokens are scored exactly
     once (this is the standard HuggingFace long-text PPL recipe).

==========================================================================
  BUGS FIXED vs. THE ORIGINAL ``CalculatePPL.ipynb``
==========================================================================

The reference notebook from https://github.com/Weihang-Huang/ALMs contained
several bugs. We fix all of them here and document each so the fix is
auditable:

  1. **``ppl_df`` was never defined** (cell 3). The original wrote
     ``ppl_df = ppl_df.parallel_apply(cal_ppl_global, axis=1)`` but never
     created ``ppl_df`` — it was meant to be the output of the preceding
     ``corpusrow2textdf`` step. The whole aggregation pipeline was therefore
     broken at runtime. Fix: ``aggregate_ppl`` builds the per-text dataframe
     explicitly (``corpus_df -> _resstr2list -> per-row text_df -> PPL``).

  2. **Pipeline between cells was broken.** The original jumped from
     ``corpus_df.parallel_apply(...)`` to a different ``ppl_df`` object with
     no connecting assignment. Fix: a single explicit loop.

  3. **Global-variable leakage under multiprocessing.** ``resstr2list`` and
     ``corpusrow2textdf`` read ``feature_categories`` as a bare global. With
     ``pandarallel``/``multiprocess`` it can be undefined in worker
     processes. Fix: pass it as an explicit ``args=``.

  4. **``CrossEntropyLoss`` re-instantiated per token.** The original created
     a new ``CrossEntropyLoss()`` inside the per-token inner loop (thousands
     of allocations per text). Fix: instantiate once with
     ``reduction="none"`` outside the loop.

  5. **Filename splitting on ``-`` broke for tags containing ``-``.** The
     original parsed ``model_tag-text_tag`` with ``split("-")``, which
     mis-split hyphenated author tags. Fix: ``_parse_pair_name`` splits on
     the *first* ``-`` only.

  6. **Per-token losses double-counted context tokens.** The original derived
     ``shift_labels`` from ``input_ids`` (all tokens) rather than from
     ``target_ids`` (masked). In the sliding-window loop the context tokens
     were meant to be masked via ``target_ids[:, :-trg_len] = -100`` so
     ``outputs.loss`` only scored new tokens — but the per-token loss
     extraction ignored that mask and re-scored the context on every
     overlapping window. Fix: derive ``shift_labels`` from ``target_ids`` and
     drop the resulting zeros so they do not dilute the PPL.

  7. **``predict_and_benchmark`` grouped by ``text_num`` alone.** Because
     ``text_num`` is reset per author inside ``aggregate_ppl``, grouping by
     ``text_num`` alone conflates author0's text #0 with author1's text #0,
     producing one prediction for two different documents. Fix: group by
     ``(true_tag, text_num)`` together.

==========================================================================
  HOW TO USE (minimal example)
==========================================================================

    from thesis_aa import config, data as data_mod
    from thesis_aa.alms import ppl as alms_ppl

    train_df, test_df = data_mod.load_natural()  # or load_benchmark
    # (assume models already trained under config.MODEL_DIR via alms.train)

    result_path = alms_ppl.score_all_pairs(
        train_df, test_df, model_dir=config.MODEL_DIR,
    )
    ppl_paths = alms_ppl.aggregate_ppl()
    bench_paths = alms_ppl.predict_and_benchmark(ppl_paths)
    # bench_paths[0] is a CSV with accuracy (macro-accuracy per author),
    # one row per "feature" (e.g. global-ppl:(losses)) plus one per author.
"""

from __future__ import annotations

import ast
import copy
import csv
import io
import math
import os
import zipfile
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel

from .. import config

# Feature categories written to the per-pair CE log CSV (CalculatePPL.ipynb).
FEATURE_CATEGORIES = [
    "tokens", "losses", "lemmas", "poss", "tags", "shapes",
    "alphas", "stops", "morphs", "deps", "ent_types",
]


# ---------------------------------------------------------------------------
# Per-text cross-entropy and (optional) on-the-fly spaCy tagging
# ---------------------------------------------------------------------------

def _spacy_pipeline():
    """Load a spaCy pipeline, preferring GPU but falling back to CPU."""
    import spacy
    try:
        spacy.require_gpu()
    except Exception:
        pass
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        try:
            return spacy.load("en_core_web_trf")
        except OSError:
            return None


def compute_ce_per_text(
    text: str,
    model: GPT2LMHeadModel,
    tokenizer,
    device: torch.device,
    stride: int = 128,
    tagger=None,
) -> dict:
    """Compute per-token negative log-likelihood for one text.

    Walks the text in overlapping windows of length ``model.config.n_positions``
    (1024 for gpt2-base), advanced by ``stride``. For each window:

      * The model produces logits at every position.
      * Position ``i`` predicts the token at position ``i+1``.
      * We compute ``-log p(token_{i+1} | context)`` — the NLL of the realised
        next token — using ``CrossEntropyLoss(reduction="none")``.
      * Context shared with the previous window is masked (``-100``) so each
        token is scored exactly once across the whole text.

    Returns a dict with keys from :data:`FEATURE_CATEGORIES`:
    ``tokens``, ``losses`` (per-token NLL, length ``len(tokens)-1`` padded to
    ``len(tokens)`` with a trailing 0.0 for alignment), and — when a spaCy
    ``tagger`` is supplied — the linguistic annotations used in
    CalculatePPL.ipynb (lemma, pos, tag, shape, alpha, stop, morph, dep,
    ent_type). When no tagger is supplied the annotation keys are filled with
    empty strings, so the downstream CSV schema stays consistent.

    **Why the sliding window?** GPT-2 cannot accept more than ``n_positions``
    tokens at once. A 2000-token document therefore needs >=2 forward passes.
    Using a ``stride`` smaller than ``max_length`` makes windows overlap so
    every token is predicted *with its full left context* up to the window
    size — which is important for accuracy. The masking via ``-100`` then
    ensures each token is only *counted* once (in the window where it is a
    "new" token), even though it may appear as context in later windows.
    """
    clean = text.replace("<BOS>", "").replace("<EOS>", "")
    encodings = tokenizer(clean, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    tokens = tokenizer.batch_decode(encodings.input_ids[0])
    losses: list[float] = []

    max_length = model.config.n_positions
    prev_end_loc = 0
    model.eval()

    loss_fct = CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100  # mask the context, score only new tokens

            outputs = model(input_ids, labels=target_ids)
            logits = outputs.logits  # (1, L, V)

            # IMPORTANT: derive shift_labels from *target_ids* (the masked
            # version), not from input_ids. The window masks context tokens
            # (target_ids[:, :-trg_len] = -100) so that ``outputs.loss`` only
            # scores the *new* tokens in the window. If we used input_ids here,
            # the context tokens would be scored again on every overlapping
            # window, double-counting their losses. Using target_ids keeps the
            # per-token loss list aligned one-to-one with the predicted
            # positions and avoids the double count.
            shift_labels = target_ids[..., 1:].contiguous()
            shift_logits = logits[..., :-1, :].contiguous()

            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            # ``reduction="none"`` returns 0.0 for the -100-masked positions
            # (CrossEntropyLoss ignores index -100 by default, emitting 0.0
            # for those positions). We must drop those zeros so they do not
            # dilute the per-token loss list and the downstream perplexity.
            mask = shift_labels.view(-1) != -100
            losses.extend(token_losses[mask].tolist())

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    # On-the-fly linguistic tagging (CalculatePPL.ipynb uses spaCy over the
    # GPT-2 token strings to keep annotations aligned with subword tokens).
    record = {"tokens": tokens, "losses": losses}
    if tagger is not None:
        try:
            import spacy.tokens
            raw_doc = spacy.tokens.Doc(tagger.vocab, words=tokens)
            doc = tagger(raw_doc)
            for attr, key in [
                ("lemma_", "lemmas"), ("pos_", "poss"), ("tag_", "tags"),
                ("shape_", "shapes"), ("is_alpha", "alphas"), ("is_stop", "stops"),
                ("morph", "morphs"), ("dep_", "deps"), ("ent_type_", "ent_types"),
            ]:
                record[key] = [getattr(tok, attr) if not attr.endswith("_") or isinstance(getattr(tok, attr), str)
                               else str(getattr(tok, attr)) for tok in doc]
        except Exception:
            for key in FEATURE_CATEGORIES[2:]:
                record[key] = [""] * len(tokens)
    else:
        for key in FEATURE_CATEGORIES[2:]:
            record[key] = [""] * len(tokens)

    return record


# ---------------------------------------------------------------------------
# Score all (model_author, text_author) pairs
# ---------------------------------------------------------------------------

def _safe_pair_name(model_tag: str, text_tag: str, sep: str = "-") -> str:
    """Construct a filename-safe pair name robust to tags containing ``-``.

    The original split on ``-``; we split on the first ``-`` only when parsing.
    """
    return f"{model_tag}{sep}{text_tag}"


def _parse_pair_name(name: str, sep: str = "-") -> tuple[str, str]:
    """Inverse of :func:`_safe_pair_name`: split on the first ``sep`` only."""
    idx = name.find(sep)
    if idx == -1:
        raise ValueError(f"Pair name '{name}' lacks separator '{sep}'")
    return name[:idx], name[idx + len(sep):]


def score_all_pairs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_dir: str = config.MODEL_DIR,
    ce_log_home: str = os.path.join(config.RESULTS_DIR, "ce_log"),
    result_path: str = os.path.join(config.RESULTS_DIR, "ppl_result.csv"),
    device: torch.device | None = None,
    tag_col: str = "author_tag",
    text_col: str = "text",
    stride: int = 128,
    use_tagger: bool = False,
    limit_texts_per_author: int | None = None,
) -> str:
    """Compute cross-entropy for every (ALM, author) pair.

    **Data flow (the "N x M" scoring matrix):**

    If there are ``n_models`` authorial models (one per candidate, in
    ``model_dir``) and ``n_authors`` distinct true authors in the test set,
    this function produces an ``n_models x n_authors`` grid of mean
    perplexities. Each cell is "how predictable is author_j's test text
    under author_i's model?". The predicted author for a test text is then
    the *column* (model) with the *lowest* PPL — that is the argmin
    attribution rule (Ch.5 Sec 5.3.2).

    **Outputs written to disk:**

    1. ``ce_log/<model_tag>-<text_tag>.csv.7z`` — one LZMA-compressed CSV per
       (model, author) pair. Each row is one test text; columns are the
       per-token losses + linguistic annotations (the FEATURE_CATEGORIES).
       These are the raw inputs to ``aggregate_ppl``.

    2. ``result_path`` (default ``results/ppl_result.csv``) — one row per
       (model, author) pair with columns: model_tag, text_tag, stride, ppl.
       This is the resumability log: pairs already in this file are skipped
       on restart (matching the original notebook's design).

    **Resumability:** If ``result_path`` already contains a (model, author)
    pair, that pair is skipped. This lets you interrupt and restart a long
    scoring run without redoing completed work.

    Returns ``result_path``.
    """
    if device is None:
        device = config.get_device()
    os.makedirs(ce_log_home, exist_ok=True)
    os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)

    model_tags = sorted(
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d))
    )
    if not model_tags:
        raise FileNotFoundError(f"No authorial models found in {model_dir}")

    text_tags = sorted(test_df[tag_col].unique().tolist())

    tagger = _spacy_pipeline() if use_tagger else None

    if not os.path.isfile(result_path):
        with open(result_path, "w", encoding="utf-8") as f:
            f.write("model_tag,text_tag,stride,ppl\n")

    logged_pairs = set()
    with open(result_path, "r", encoding="utf-8") as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                logged_pairs.add((parts[0], parts[1]))

    for model_tag in model_tags:
        model_id = os.path.join(model_dir, model_tag)
        model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        for text_tag in text_tags:
            if (model_tag, text_tag) in logged_pairs:
                print(f"[ALMs/PPL] skip {model_tag} x {text_tag} (logged)")
                continue
            print(f"[ALMs/PPL] {model_tag} x {text_tag}")

            sub = test_df[test_df[tag_col] == text_tag]
            if limit_texts_per_author is not None:
                sub = sub.head(limit_texts_per_author)
            nlls: list[float] = []
            records = []
            for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"{model_tag}/{text_tag}"):
                rec = compute_ce_per_text(row[text_col], model, tokenizer, device,
                                          stride=stride, tagger=tagger)
                records.append([rec[k] for k in FEATURE_CATEGORIES])
                # text-level NLL = mean of per-token losses
                if rec["losses"]:
                    nlls.append(sum(rec["losses"]) / len(rec["losses"]))

            # Write per-pair CE log (7z + LZMA, matching original).
            pair_name = _safe_pair_name(model_tag, text_tag)
            ce_fn = pair_name + ".csv"
            ce_zip = ce_fn + ".7z"
            with zipfile.ZipFile(os.path.join(ce_log_home, ce_zip), "w",
                                 compression=zipfile.ZIP_LZMA) as z:
                with z.open(ce_fn, "w") as f:
                    tw = io.TextIOWrapper(f, "utf-8")
                    csvw = csv.writer(tw)
                    csvw.writerows(records)
                    tw.flush()

            ppl = math.exp(sum(nlls) / len(nlls)) if nlls else float("inf")
            with open(result_path, "a", encoding="utf-8") as f:
                f.write(f"{model_tag},{text_tag},{stride},{ppl}\n")
            print(f"[ALMs/PPL] {model_tag} x {text_tag}: PPL={ppl:.2f}")

        del model
        # Free the (potentially large) ALM before loading the next one.
        # CUDA has a cache; xpu/CPU don't, but calling the API when
        # available keeps memory pressure low on long multi-model runs.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result_path


# ---------------------------------------------------------------------------
# Aggregate per-text PPL (fixed pipeline)
# ---------------------------------------------------------------------------

def _resstr2list(item: pd.Series,
                 feature_categories: list[str]) -> pd.Series:
    """Parse stringified lists in a CE-log row.

    When a CE-log CSV is read back with ``pd.read_csv`` the list-valued cells
    (tokens, losses, lemmas, ...) are stored as *strings* like
    ``"['the', 'castle', ...]"``. We recover the actual Python lists with
    ``ast.literal_eval``.

    **The losses alignment convention (read this twice).** A causal LM
    predicts token ``i+1`` from the tokens before it, so a text of ``n``
    tokens yields exactly ``n-1`` per-token losses: ``losses[j]`` is the NLL
    of ``tokens[j+1]``. The first token has no left context and is never
    scored. For storage convenience we *pad* the losses list with a trailing
    ``0.0`` so it has the same length as ``tokens`` and can be zipped with
    it 1:1; ``losses_shifted`` then shifts that padded list right by one so
    that entry ``j`` aligns with ``tokens[j]``. Downstream PPL averages
    therefore include one constant 0.0 term — a deliberate quirk inherited
    from the reference notebook's convention.

    Fix #3: ``feature_categories`` is passed as an argument (via
    ``DataFrame.apply(..., args=...)``) rather than read as a global, so it
    survives under multiprocessing.
    """
    for cat in feature_categories:
        val = ast.literal_eval(str(item[cat]))
        item[cat] = val
        item[cat + "_count"] = len(val)
    # losses are aligned to tokens[:-1]; pad to len(tokens).
    if item["losses_count"] != item["tokens_count"] - 1:
        item["losses"] = item["losses"][: item["tokens_count"] - 1]
    item["losses"] = item["losses"] + [0.0]
    item["losses_count"] = len(item["losses"])
    item["losses_shifted"] = [0.0] + item["losses"][:-1]
    item["losses_shifted_count"] = len(item["losses_shifted"])
    return item


def aggregate_ppl(
    ce_log_home: str = os.path.join(config.RESULTS_DIR, "ce_log"),
    out_home: str = os.path.join(config.RESULTS_DIR, "ppl_dfs_buffer"),
) -> list[str]:
    """Aggregate per-text PPL for each (model, author) pair.

    Fix #1 & #2: the original ``CalculatePPL.ipynb`` referenced ``ppl_df``
    without constructing it. Here the full chain is explicit:
      corpus_df -> _resstr2list -> per-row PPL aggregation -> ppl_df

    Each ``.csv.7z`` pair log is LZMA-compressed and its cells hold
    stringified Python lists, so parsing is expensive; each log is parsed
    exactly once.

    The output is a single tidy CSV (``ppl_dfs_buffer.csv``) with one row per
    (test text, candidate ALM) and one ``global-ppl:(*)`` column per feature.
    """
    os.makedirs(out_home, exist_ok=True)

    fps = [
        os.path.join(ce_log_home, f)
        for f in sorted(os.listdir(ce_log_home))
        if f.endswith(".csv.7z")
    ]
    if not fps:
        raise FileNotFoundError(f"No *.csv.7z CE logs in {ce_log_home}")

    # Parse (model_tag, text_tag) from filenames using robust first-'-' split.
    tasks = []
    for fp in fps:
        base = os.path.basename(fp).split(".csv.7z")[0]
        m, t = _parse_pair_name(base)
        tasks.append((m, t))

    rows = []
    for (model_tag, text_tag) in tqdm(tasks, desc="parse CE logs"):
        fp = os.path.join(ce_log_home, _safe_pair_name(model_tag, text_tag) + ".csv.7z")
        with zipfile.ZipFile(fp, "r") as z:
            inner = [n for n in z.namelist() if n.endswith(".csv")][0]
            with z.open(inner) as f:
                corpus_df = pd.read_csv(f, names=FEATURE_CATEGORIES, compression=None)

        ppl_rows = []
        for _, row in corpus_df.apply(
            _resstr2list, axis=1,
            args=(copy.deepcopy(FEATURE_CATEGORIES),),
        ).iterrows():
            # Compute the two global-PPL features from the per-token losses.
            ppl_row = row.copy()
            for aim in ("losses_shifted", "losses"):
                losses_list = list(ppl_row[aim])
                ppl_row[f"global-ppl:({aim})"] = (
                    math.exp(sum(losses_list) / len(losses_list)) if losses_list else 0.0
                )
            ppl_row["candidate_tag"] = model_tag
            ppl_row["true_tag"] = text_tag
            # text_num is unique only *within* a (candidate, true-author)
            # pair — predict_and_benchmark must group by both keys.
            ppl_row["text_num"] = len(ppl_rows)
            ppl_rows.append(ppl_row)

        rows.extend(ppl_rows)

    out_path = os.path.join(out_home, "ppl_dfs_buffer.csv")
    ppl_dfs = pd.DataFrame(rows)
    cols = [c for c in ppl_dfs.columns if c.startswith("global-ppl:")]
    ppl_dfs = ppl_dfs[["true_tag", "candidate_tag", "text_num"] + cols]
    ppl_dfs.to_csv(out_path, index=False)
    print(f"[ALMs/PPL] wrote {out_path}")

    return [out_path]


# ---------------------------------------------------------------------------
# Predictions and benchmarking (mirrors CalculatePPL.ipynb)
# ---------------------------------------------------------------------------

def predict_and_benchmark(
    ppl_dfs_paths: list[str],
    pred_home: str = os.path.join(config.RESULTS_DIR, "pred_df_buffer"),
    benchmark_home: str = os.path.join(config.RESULTS_DIR, "benchmark_results_df_home"),
) -> list[str]:
    """Convert per-text PPL buffers into predictions and benchmark metrics.

    For each PPL buffer CSV, for each ``global-ppl:(*)`` feature, attribute
    each test text to the candidate with the lowest PPL (argmin). Then compute
    accuracy globally and per true author (the manuscript's reported metric).
    """
    from ..eval import build_benchmark_results_df

    os.makedirs(pred_home, exist_ok=True)
    os.makedirs(benchmark_home, exist_ok=True)
    bench_paths = []

    for ppl_path in ppl_dfs_paths:
        pred_path = os.path.join(pred_home, "pred_df_buffer.csv")
        bench_path = os.path.join(benchmark_home, "benchmark_results_df_buffer.csv")
        bench_paths.append(bench_path)

        ppl_df = pd.read_csv(ppl_path)
        by_features = [c for c in ppl_df.columns if c.startswith("global-ppl:")]

        # IMPORTANT: group by (true_tag, text_num) together, not text_num alone.
        # ``text_num`` is only unique *within* a true-author partition (it is
        # reset per author in ``aggregate_ppl``). Grouping by ``text_num`` alone
        # would conflate author0's text #0 with author1's text #0, producing a
        # single prediction for two different documents and corrupting both the
        # accuracy and the count of evaluated documents.
        pred_rows = []
        for by_feature in by_features:
            for (true_tag, text_num), batch in ppl_df.groupby(["true_tag", "text_num"]):
                feat_vec = batch.set_index("candidate_tag")[by_feature]
                pred_tag = feat_vec.idxmin()
                pred_rows.append({
                    "by": by_feature,
                    "true_tag": true_tag,
                    "text_num": text_num,
                    "pred_tag": pred_tag,
                })

        pred_df = pd.DataFrame(pred_rows)
        pred_df.to_csv(pred_path, index=False)

        bench_df = build_benchmark_results_df(pred_df)
        bench_df.to_csv(bench_path, index=False)
        print(f"[ALMs/PPL] benchmark -> {bench_path}")

    return bench_paths


# ---------------------------------------------------------------------------
# Token-level comparative NLL (Ch.5 Sec 5.3.3, Eq. 3-4)
# ---------------------------------------------------------------------------

def compute_cnll(
    nll_matrix: np.ndarray | list[list[float]],
    authors: Sequence[str],
    candidate: str | None = None,
) -> np.ndarray:
    """Comparative Negative Log-Likelihood (Ch.5 Eq. 3-4).

    **What it means (intuition):** A single token's NLL under one author's
    model is hard to interpret on its own (is 2.3 high? low?). But if you
    subtract the *average* NLL that the *other* candidate models gave the
    same token, you get a signed value that says "this token is more
    predictable for this candidate than for the others" (negative) or "less
    predictable" (positive). Summing CNLL over a span tells you which parts
    of the questioned document drove the attribution toward (or away from)
    a candidate. This is the interpretability feature Ch.5 Sec 5.3.3.

    **Inputs:**
      ``nll_matrix``  — shape ``(n_tokens, n_authors)``. Each *row* is one
        token; each *column* is one candidate's ALM. The value at ``[i, j]``
        is the NLL of token ``i`` under author ``j``'s model.
      ``authors``     — the column labels, in the same order as the columns.
      ``candidate``   — which author's column to compare. If ``None``, the
        author with the lowest *total* NLL (i.e. the predicted author) is
        used — this matches how the thesis uses CNLL for the attributed
        author.

    **Output:** a ``(n_tokens,)`` array. ``CNLL[i] < 0`` => token ``i``
    favours the candidate; ``> 0`` => it disfavours her.

    **Pairwise form:** When there are only two candidates ``a, b`` this
    reduces to ``CNLL(a, i) = NLL_a(i) - NLL_b(i)`` (Eq. 3). The n-author
    form (Eq. 4) compares one candidate to the *mean* of the remaining
    ``n-1`` candidates, which is what this function implements.
    """
    M = np.asarray(nll_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("nll_matrix must be 2-D (n_tokens, n_authors)")
    n_tokens, n_authors = M.shape
    if n_authors < 2:
        raise ValueError("CNLL requires at least 2 candidate authors")

    if candidate is None:
        # Predicted author = argmin total NLL.
        candidate_idx = int(np.argmin(M.sum(axis=0)))
    else:
        candidate_idx = list(authors).index(candidate)

    others = [j for j in range(n_authors) if j != candidate_idx]
    other_mean = M[:, others].mean(axis=1)
    return M[:, candidate_idx] - other_mean