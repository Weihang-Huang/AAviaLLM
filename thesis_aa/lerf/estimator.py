"""LERF estimator and standard relative-frequency estimators (Ch.6).

==========================================================================
  WHAT THIS MODULE DOES (big picture for newcomers)
==========================================================================

This module estimates "how often each word type appears" in a corpus — but
instead of just *counting* (the way traditional corpus linguistics does), it
uses a *frozen GPT-2* to produce a smarter estimate that also covers words
the corpus never actually contains. That smarter estimate is called **LERF**
(LLM-Estimated Relative Frequency).

Why does this matter? In corpus linguistics you often want to know the true
relative frequency of every word in some variety of language. You only have
a *sample* corpus, so rare words may be absent just by chance. Traditional
estimators (Add-One, Good-Turing, etc.) try to patch this using only the
counts in the sample. LERF instead asks the LLM: "given the contexts in this
corpus, what word would you predict?" and sums those predictions across all
positions. Because the LLM has seen vast amounts of text during pretraining,
it "knows" about words the sample corpus never shows — and it can give them
non-zero probability. The thesis (Ch.6) shows LERF is usually more accurate
than the five classical estimators.

This module provides:
  - ``lerf_estimate``           — the LERF estimator (Ch.6 Sec 6.4);
  - ``mle_estimate``            — the baseline: just count and divide;
  - ``standard_estimators``     — the five classical estimators for comparison
                                  (Add-One, Good-Turing, Katz-Backoff,
                                  Kneser-Ney, Witten-Bell; Ch.6 Sec 6.3.1);
  - ``lmse``                    — the goodness-of-fit metric (Ch.6 Sec 6.5.4);
  - ``split_corpus``            — sample-vs-reference split (Ch.6 Sec 6.5.2);
  - ``evaluate_all_estimators`` — run the full comparison on one corpus.

==========================================================================
  KEY CONCEPT (LERF in one paragraph)
==========================================================================

Given a corpus of texts, for each text we feed every prefix to the (frozen)
LLM and read out its full next-token probability distribution over the
50,257-token GPT-2 vocabulary. We sum those distributions across *every
position* in the corpus, then normalise so the vector sums to 1. That vector
is the LERF estimate:

    p_lerf(v) = (1/Z) * sum over positions i of  p_model(v | context_i)

where ``Z`` = number of positions (each distribution already sums to 1, so
the sum of all positions' distributions sums to Z, and dividing by Z gives a
probability distribution). Crucially, types that never occur in the corpus
but that the LLM considers plausible in some context receive *non-zero*
mass — that is the whole point of LERF (Ch.6 Sec 6.4).

==========================================================================
  THE FIVE STANDARD ESTIMATORS (Ch.6 Sec 6.3.1) — quick intuition
==========================================================================

All five take the *observed counts* ``c(w)`` and try to redistribute
probability mass better than raw MLE (which gives unseen types 0).

  **Add-One (Laplace)** — add 1 to every count: ``p = (c+1)/(N+V)``.
    Simple, but over-smooths rare words and gives unseen types the same
    mass as once-seen types.

  **Good-Turing** — re-estimate each count ``c`` as ``c* = (c+1) N_{c+1}/N_c``
    using the *frequency-of-frequencies* ``N_r`` (how many types occur exactly
    ``r`` times). The total mass of unseen types is estimated as ``N_1/N``
    and spread across all ``V - T`` unseen types. Intuition: the proportion
    of tokens you'll see a *new* type next is roughly the proportion of types
    you've only seen once.

  **Katz-Backoff** — discount observed counts by a Good-Turing factor, then
    redistribute the freed mass to unseen types (a simplified unigram form of
    the classic Katz backoff). This implementation is the simplified
    unigram version; the full Katz model uses higher-order context.

  **Kneser-Ney** — subtract a fixed discount ``D`` (=0.75 here) from each
    observed count and redistribute the total discounted mass uniformly. This
    is the simplified unigram form; the classic KN uses continuation
    probabilities from higher orders. KN tends to over-smooth in the thesis
    results (Ch.6).

  **Witten-Bell** — ``p(w) = c(w)/(N+T)`` for seen types,
    ``T/((V-T)(N+T))`` for unseen, where ``T`` = number of distinct observed
    types. Intuition: the probability of seeing something new is proportional
    to how many *distinct* things you've already seen.

  All five return a length-``V`` array summing to 1.

==========================================================================
  GOODNESS-OF-FIT: LMSE (Ch.6 Sec 6.5.4, Eq. 6.9)
==========================================================================

LMSE = Log Mean Squared Error. Following the thesis: compute the mean of
the squared differences between estimated and ground-truth *raw* relative
frequencies across all word types observed in the reference corpus (Sec
6.5.2 defines the target vocabulary as the types observed in the reference
corpus), then take the base-2 logarithm::

    LMSE = log2( mean( (p_hat - p_ref)^2 ) )

Lower (more negative) values indicate a better fit (Table 6.2: "Lower
(more negative) values indicate superior performance"); a perfect fit
gives ``-inf``. Typical values on the thesis's natural-English corpora
are around -24 to -31. We compare raw frequencies (not log-probabilities)
because Eq. 6.9 squares the raw residuals; the single log2 compresses the
scale so "a difference of one unit corresponds to a halving or doubling of
the underlying mean squared error" (Sec 6.5.4).

==========================================================================
  HOW TO USE (minimal example)
==========================================================================

    from thesis_aa import data as data_mod
    from thesis_aa.lerf import estimator as lerf_est

    train_df, _ = data_mod.load_natural()
    row = lerf_est.evaluate_all_estimators(train_df, model_name='gpt2')
    # row has one row, columns: LERF, MLE, Add-One, Good-Turing,
    # Katz-Backoff, Kneser-Ney, Witten-Bell — each an LMSE score.
    print(row.T)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .. import config

VOCAB_SIZE = config.GPT2_VOCAB_SIZE


# ---------------------------------------------------------------------------
# Shared token counting (used by MLE, standard estimators, and MFW selection)
# ---------------------------------------------------------------------------

def count_token_ids(
    texts: list[str] | pd.Series,
    tokenizer=None,
    vocab_size: int = VOCAB_SIZE,
) -> np.ndarray:
    """Count GPT-2 token-id occurrences across ``texts``.

    This is the shared substrate of three callers that previously each had
    their own copy of the same loop (MLE, the five classical estimators,
    and LERF-AA's MFW selection): strip the ``<BOS>``/``<EOS>`` markers,
    tokenize each text, and tally token ids into a length-``vocab_size``
    vector. Returns the raw integer counts (NOT normalised).
    """
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    counts = np.zeros(vocab_size, dtype=np.float64)
    for text in texts:
        clean = str(text).replace("<BOS>", "").replace("<EOS>", "")
        if not clean.strip():
            continue
        ids = tokenizer(clean, return_tensors="pt").input_ids[0].tolist()
        for tid in ids:
            if 0 <= tid < vocab_size:
                counts[tid] += 1
    return counts


# ---------------------------------------------------------------------------
# LERF estimation (Ch.6 Sec 6.4.3)
# ---------------------------------------------------------------------------

def _lerf_windows(seq_len: int, max_len: int, stride: int) -> list[tuple[int, int, int, int]]:
    """Plan the overlapping scoring windows for one text (pure helper).

    Returns a list of ``(begin, end, first_row, last_row)`` tuples. Each
    window covers token indices ``[begin, end)``; its logits row ``j``
    (absolute position ``begin + j``) predicts token ``begin + j + 1``.
    The caller sums the softmax distributions of rows
    ``[first_row, last_row]`` (inclusive) of that window.

    Guarantees (Ch.6 Sec 6.4.3 — a text of ``n`` tokens yields ``n-1``
    contexts, each conditioned on its preceding sequence):

    1. **Exact coverage.** Every predicting position ``0 .. seq_len-2`` is
       counted in exactly one window — never zero, never twice.
    2. **Context floor.** A counted position always sees at least
       ``max_len - stride`` tokens of left context (and at most
       ``max_len - 1``); positions in the first window see their full true
       prefix. Exact maximal context for *every* position is impossible
       with any stride > 1 (adjacent positions would need window starts
       one token apart), so the overlap size — ``max_len - stride`` — is
       the context guarantee; the default ``stride = max_len // 2``
       guarantees half the model context, the standard HuggingFace
       long-text recipe. Pass a smaller ``stride`` for a higher floor at
       more compute.
    3. **No wasted rows.** Non-final windows count through their last row
       (which predicts the next window's first token with ``max_len - 1``
       tokens of context); final windows exclude the row whose target
       would fall outside the text.
    """
    if seq_len < 2:
        return []
    if seq_len <= max_len:
        # One window: rows 0..seq_len-2 predict tokens 1..seq_len-1.
        return [(0, seq_len, 0, seq_len - 2)]

    # Clamp: stride must be <= max_len - 1 so consecutive windows overlap
    # (a stride equal to max_len would drop ALL context at boundaries).
    eff_stride = max(1, min(stride, max_len - 1))
    windows: list[tuple[int, int, int, int]] = []
    prev_last_counted = -1  # highest absolute position counted so far
    begin = 0
    while prev_last_counted < seq_len - 2:
        end = min(begin + max_len, seq_len)
        first_row = max(prev_last_counted + 1 - begin, 0)
        if end < seq_len:
            # Non-final window: the last row (position end-1) predicts
            # token `end`, which exists later in the text — count it here
            # with its full max_len - 1 tokens of context.
            last_row = (end - begin) - 1
        else:
            # Final window: the last row would predict token `seq_len`,
            # which does not exist — exclude it.
            last_row = (end - begin) - 2
        if first_row <= last_row:
            windows.append((begin, end, first_row, last_row))
            prev_last_counted = begin + last_row
        if end == seq_len:
            break
        begin += eff_stride
    return windows


def lerf_estimate(
    texts: list[str] | pd.Series,
    model_name: str = "gpt2",
    model=None,
    tokenizer=None,
    device: torch.device | None = None,
    stride: int | None = None,
) -> np.ndarray:
    """Estimate vocabulary-wide relative frequencies via a frozen causal LLM.

    For every context position in ``texts``, the model produces a distribution
    over the full vocabulary (``V = 50,257`` for GPT-2). These distributions
    are summed across all positions in the corpus and normalised:

        p_lerf(v) = (1/Z) * sum_i  p_model(v | x_{<i})

    where ``Z`` is the total probability mass accumulated (``= sum over
    positions`` because each distribution sums to 1, so ``Z`` equals the number
    of context positions). Types that never occur but receive probability from
    the model obtain non-zero estimates (Ch.6 Sec 6.4).

    **Context windowing (Ch.6 Sec 6.4.3):** the manuscript extracts, for a
    text of ``n`` tokens, ``n-1`` contexts — "the sequence of words that
    precedes" each token — i.e. every position is conditioned on its
    preceding sequence. GPT-2's fixed context (``n_positions`` = 1024 for
    GPT-2) cannot hold a longer document at once, so long texts are scored
    in overlapping windows (window size = ``n_positions``, advanced by
    ``stride``; default half the context, the standard HuggingFace
    long-text recipe). Each predicting position is counted **exactly
    once**, and a counted position always sees at least
    ``n_positions - stride`` tokens of left context (up to
    ``n_positions - 1``); exact maximal context for every position is
    impossible with any stride > 1, so the overlap size is the context
    guarantee (see ``_lerf_windows``). Reporting the windowing follows the
    manuscript's Ch.3 requirement that any sliding-window procedure be
    reported. With a stride equal to the model context (the previous
    behaviour), every token after each chunk boundary lost ALL prior
    context — a context floor of zero — contradicting the Sec 6.4.3 design.

    Returns a 1-D array of length ``V`` summing to 1.
    """
    if device is None:
        device = config.get_device()

    if model is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = model.config.vocab_size
    acc = torch.zeros(vocab_size, dtype=torch.float64, device=device)

    max_len = model.config.n_positions
    # Default stride: half the context — the standard HuggingFace long-text
    # recipe. Every counted position then sees >= n_positions/2 tokens of
    # left context (up to n_positions - 1). See _lerf_windows for the exact
    # coverage and context-floor guarantees.
    eff_stride = (max_len // 2) if stride is None else max(1, min(stride, max_len - 1))

    for text in tqdm(texts, desc="LERF"):
        clean = str(text).replace("<BOS>", "").replace("<EOS>", "")
        if not clean.strip():
            continue
        enc = tokenizer(clean, return_tensors="pt")
        ids = enc.input_ids[0]
        seq_len = ids.size(0)
        if seq_len < 2:
            continue

        for begin, end, first_row, last_row in _lerf_windows(seq_len, max_len, eff_stride):
            chunk = ids[begin:end].unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(chunk).logits  # (1, L, V)
                probs = F.softmax(logits, dim=-1)  # (1, L, V)
            acc += probs[0, first_row:last_row + 1, :].sum(dim=0).double()

    total = acc.sum()
    if total.item() == 0:
        return np.zeros(vocab_size, dtype=np.float64)
    p = (acc / total).cpu().numpy()
    return p.astype(np.float64)


# ---------------------------------------------------------------------------
# Observed (MLE) estimate
# ---------------------------------------------------------------------------

def mle_estimate(texts: list[str] | pd.Series, tokenizer=None,
                 model_name: str = "gpt2", vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    """Maximum-likelihood (observed) relative-frequency estimate.

    Counts each token id and divides by the total. Unseen types get 0.
    """
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    counts = count_token_ids(texts, tokenizer=tokenizer, vocab_size=vocab_size)
    total = counts.sum()
    if total == 0:
        return counts
    return counts / total


# ---------------------------------------------------------------------------
# Standard estimators (Ch.6 Sec 6.3.1)
# ---------------------------------------------------------------------------

def standard_estimators(
    texts: list[str] | pd.Series,
    tokenizer=None,
    model_name: str = "gpt2",
    vocab_size: int = VOCAB_SIZE,
) -> Dict[str, np.ndarray]:
    """Compute the five standard estimators from observed counts.

    Returns a dict keyed by estimator name, each a length-``vocab_size`` array
    summing to 1: Add-One, Good-Turing, Katz-Backoff, Kneser-Ney, Witten-Bell.
    """
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Observed counts and unigram counts-of-counts (shared counting helper).
    counts = count_token_ids(texts, tokenizer=tokenizer, vocab_size=vocab_size)

    N = counts.sum()
    return {
        "Add-One": _add_one(counts, vocab_size),
        "Good-Turing": _good_turing(counts, vocab_size),
        "Katz-Backoff": _katz_backoff(counts, vocab_size),
        "Kneser-Ney": _kneser_ney(counts, vocab_size),
        "Witten-Bell": _witten_bell(counts, vocab_size),
    }


def _add_one(counts: np.ndarray, vocab_size: int) -> np.ndarray:
    """Laplace (Add-One) smoothing: (c+1)/(N+V)."""
    return (counts + 1.0) / (counts.sum() + vocab_size)


def _good_turing(counts: np.ndarray, vocab_size: int) -> np.ndarray:
    """Simple Good-Turing (Gale & Sampson, 1995).

    Adjusts each observed count ``c`` to ``c* = (c+1) * N_{c+1} / N_c`` using
    the frequency-of-frequencies ``N_r`` (the number of types that occur
    exactly ``r`` times). Types with count 0 — the unseen vocabulary — receive
    the total mass ``N_1 / N`` (the Good-Turing estimate of the probability
    of encountering a previously-unseen type), distributed uniformly across all
    ``V - T`` unseen types (where ``T`` is the number of observed types and
    ``V`` is the vocabulary size). This is the standard treatment described in
    Gale & Sampson (1995) and referenced in Ch.6 Sec 6.3.1.
    """
    N = counts.sum()
    if N == 0:
        return np.zeros(vocab_size, dtype=np.float64)

    # Frequency of frequencies: Nr = #{types with count exactly r}.
    freq_of_freq = np.bincount(counts.astype(int))
    # N0 is not directly given by bincount; it is V - T (unseen types).
    T = (counts > 0).sum()              # number of distinct observed types
    N0 = vocab_size - T                  # number of unseen types
    N1 = freq_of_freq[1] if len(freq_of_freq) > 1 else 0

    # Adjust observed counts: c* = (c+1) * N_{c+1} / N_c.
    adjusted = counts.astype(float).copy()
    for r in range(1, len(freq_of_freq)):
        Nr = freq_of_freq[r]
        Nr1 = freq_of_freq[r + 1] if r + 1 < len(freq_of_freq) else 0
        if Nr > 0 and Nr1 > 0:
            adjusted[counts == r] = (r + 1) * Nr1 / Nr

    # Build the probability distribution.
    p = np.zeros(vocab_size, dtype=np.float64)
    seen = counts > 0
    # Total adjusted mass of seen types = sum of c* over seen types.
    seen_mass = adjusted[seen].sum()
    # Total mass reserved for unseen types = N1 / N (Good-Turing estimate).
    unseen_mass = (N1 / N) if N > 0 else 0.0
    # Normalise: ensure the seen and unseen masses sum to 1. If the
    # bookkeeping leaves a small gap or overshoot, rescale the seen mass to
    # ``1 - unseen_mass`` so the distribution sums to exactly 1.
    if unseen_mass > 1.0:
        unseen_mass = 1.0
    if seen_mass > 0:
        p[seen] = adjusted[seen] / seen_mass * (1.0 - unseen_mass)
    if N0 > 0 and unseen_mass > 0:
        p[~seen] = unseen_mass / N0
    return p


def _katz_backoff(counts: np.ndarray, vocab_size: int) -> np.ndarray:
    """Katz backoff approximation using Good-Turing discounting (Ch.6 Sec 6.3.1, Eq. 6.4).

    **Intuition:** Take a little probability mass away from every *seen* type
    (proportional to a Good-Turing discount factor) and distribute the
    collected "freed" mass uniformly across all *unseen* types. This is a
    simplified *unigram* form of Katz backoff; the full model backs off to
    lower-order n-gram distributions when a higher-order count is zero, but
    here we only have unigram counts so we back off to a uniform
    distribution over the unseen vocabulary.

    The Good-Turing discount factor for a count ``r`` is
    ``r* / r = (r+1) * N_{r+1} / (r * N_r)``, computed from the
    frequency-of-frequencies. Counts for which the factor cannot be
    estimated (``N_r`` or ``N_{r+1}`` zero) keep factor 1.0 (no discounting).

    **Normalisation (Eq. 6.4's ``lambda``):** the manuscript defines a
    "normalization factor that distributes the freed probability over the
    backed-off distribution". Two practical guards follow from this:

    * When the frequency-of-frequencies curve is non-monotone (``N_{r+1}/N_r``
      implies a discount factor > 1), the raw computation would *add* mass;
      the freed mass is clamped at >= 0 (a discount never increases mass).
    * The final distribution is renormalised to sum to exactly 1, matching
      Eq. 6.4's normalisation and the other estimators' contract.
    """
    N = counts.sum()
    if N == 0:
        return np.zeros(vocab_size, dtype=np.float64)

    freq_of_freq = np.bincount(counts.astype(int))
    gt_factor = np.ones(len(freq_of_freq), dtype=float)
    for r in range(1, len(freq_of_freq) - 1):
        Nr = freq_of_freq[r]
        Nr1 = freq_of_freq[r + 1]
        if Nr > 0 and Nr1 > 0:
            gt_factor[r] = ((r + 1) * Nr1) / (r * Nr)

    # A discount factor > 1 would inflate rather than discount; clamp so
    # freed mass is never negative (Eq. 6.4 discounts only).
    gt_factor_clamped = np.minimum(gt_factor, 1.0)

    discounted = np.zeros_like(counts, dtype=float)
    for r in range(1, len(freq_of_freq)):
        if freq_of_freq[r] > 0:
            discounted[counts == r] = counts[counts == r] * gt_factor_clamped[r]

    freed_mass = (counts * (1 - np.where(counts > 0, gt_factor_clamped[counts.astype(int)], 1.0))).sum()
    freed_mass = max(freed_mass, 0.0)
    n_unseen = (counts == 0).sum()

    p = np.zeros_like(counts, dtype=float)
    if N > 0:
        p += discounted / N
        if n_unseen > 0 and freed_mass > 0:
            p[counts == 0] = freed_mass / n_unseen / N
    total = p.sum()
    if total > 0:
        p = p / total  # Eq. 6.4 normalisation: the estimate sums to 1
    return p


def _kneser_ney(counts: np.ndarray, vocab_size: int) -> np.ndarray:
    """Simplified Kneser-Ney unigram smoothing (Ch.6 Sec 6.3.1).

    **Intuition:** Subtract a fixed discount ``D`` (=0.75, a commonly used
    value) from every observed count and spread the total subtracted mass
    uniformly across the whole vocabulary. The classical Kneser-Ney uses a
    *continuation probability* (how many distinct *preceding* contexts a type
    appeared in) as the backoff distribution; because we only have unigram
    counts here, we approximate the continuation with a uniform share of the
    discounted mass. This is why the thesis (Ch.6) finds KN tends to
    over-smooth the distribution — the unigram simplification loses KN's
    main strength (its context-aware continuation counts).
    """
    D = 0.75
    N = counts.sum()
    if N == 0:
        return np.zeros(vocab_size, dtype=np.float64)
    n_nonzero = (counts > 0).sum()
    discounted = np.maximum(counts - D, 0.0)
    lambda_ = (D * n_nonzero) / N
    p = discounted / N + lambda_ / vocab_size
    return p / p.sum()  # renormalise


def _witten_bell(counts: np.ndarray, vocab_size: int) -> np.ndarray:
    """Witten-Bell unigram smoothing.

    p(w) = c(w)/(N + T) if c(w) > 0, else T/((V-T)(N+T)),
    where T is the number of distinct observed types.
    """
    N = counts.sum()
    T = (counts > 0).sum()
    if N == 0:
        return np.zeros(vocab_size, dtype=np.float64)
    p = np.zeros(vocab_size, dtype=float)
    seen = counts > 0
    p[seen] = counts[seen] / (N + T)
    if (vocab_size - T) > 0:
        p[~seen] = T / ((vocab_size - T) * (N + T))
    return p


# ---------------------------------------------------------------------------
# Goodness-of-fit metric: LMSE (Ch.6 Sec 6.5.4)
# ---------------------------------------------------------------------------

def lmse(estimated: np.ndarray, reference: np.ndarray) -> float:
    """Log Mean Squared Error between estimated and reference distributions
    (Ch.6 Sec 6.5.4, Eq. 6.9).

    Following the thesis exactly: take the mean of the squared differences
    between estimated and ground-truth **raw relative frequencies** across
    all word types over which the estimator is evaluated, then take the
    base-2 logarithm of the result::

        LMSE = log2( mean_over_T( (p_hat_w - p_ref_w)^2 ) )

    where ``T`` is the set of types observed in the reference corpus
    (Ch.6 Sec 6.5.2 defines the target vocabulary this way — the reference
    corpus always contains the evaluation corpus, so every evaluation type
    is included). Lower (more negative) values indicate a better fit, per
    Table 6.2 ("Lower (more negative) values indicate superior
    performance"); a perfect fit gives ``-inf`` (log2 of 0). No eps floor
    and no log-of-probabilities are used — Eq. 6.9 squares the raw
    frequency residuals, and the log is applied once, to the mean.

    Example::

        # thesis-scale values: identical distributions -> -inf;
        # realistic fits on natural-English corpora land around -24..-31
        # (Table 6.2).
    """
    est = np.asarray(estimated, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if est.shape != ref.shape:
        raise ValueError(
            f"Shape mismatch: estimated {est.shape} vs reference {ref.shape}"
        )
    # Evaluate over the types observed in the reference corpus (Sec 6.5.2:
    # "the target vocabulary is defined by the set of word types observed in
    # the reference corpus").
    mask = ref > 0
    if not mask.any():
        return float("-inf")
    diffs = est[mask] - ref[mask]
    mse = float(np.mean(diffs ** 2))
    if mse == 0.0:
        return float("-inf")
    return float(np.log2(mse))


# ---------------------------------------------------------------------------
# Corpus splitting (Ch.6 Sec 6.5.2)
# ---------------------------------------------------------------------------

def split_corpus(df: pd.DataFrame, eval_frac: float = 0.5, seed: int = 0,
                 text_col: str = "text", tag_col: str = "author_tag") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a corpus into an evaluation (sample) subset and its reference set.

    Follows the manuscript's evaluation design (Ch.6 Sec 6.5.2) exactly: the
    *reference* corpus is the full input corpus (treated as the population
    whose word probabilities we want to estimate), and the *evaluation*
    corpus is a smaller sample of texts "drawn at random from that
    reference corpus" — so, as the manuscript specifies, "the reference
    corpus always contains the evaluation corpus as well as additional
    texts". The intuition is a lab experiment: you get to *see* only the
    small sample (the eval set) and must estimate the word-frequency table
    of the whole variety; the reference set is the "answer key" you score
    against.

    The default ``eval_frac=0.5`` follows the manuscript (Sec 6.5.2): "We
    have therefore chosen to draw an evaluation corpus that consists of 50%
    of the word tokens found in each reference corpus." The draw is by whole
    texts, which approximates the 50% token share.

    The sample is a plain row-level draw (NOT per-author stratified) because
    the LERF question is about the corpus as one variety of language, not
    about per-author balance.

    Returns ``(eval_df, reference_df)`` where ``reference_df`` is the full
    input corpus (index-reset) and ``eval_df`` is a random subset of it.
    """
    if not 0 < eval_frac <= 1:
        raise ValueError(
            f"eval_frac must be in (0, 1] (a fraction of texts drawn from "
            f"the reference corpus); got {eval_frac}"
        )
    reference_df = df.reset_index(drop=True)
    eval_df = df.sample(frac=eval_frac, random_state=seed).reset_index(drop=True)
    return eval_df, reference_df


# ---------------------------------------------------------------------------
# Full estimator evaluation (Ch.6 Sec 6.5)
# ---------------------------------------------------------------------------

def evaluate_all_estimators(
    df: pd.DataFrame,
    model_name: str = "gpt2",
    eval_frac: float = 0.5,
    seed: int = 0,
    device: torch.device | None = None,
    text_col: str = "text",
) -> pd.DataFrame:
    """Evaluate LERF + all standard estimators on one corpus split.

    Follows the manuscript's evaluation design (Ch.6 Sec 6.5): the input
    corpus is the reference corpus (the "population"); a random subset of
    ``eval_frac`` of its texts (default 50%, per Sec 6.5.2) forms the
    evaluation (sample) corpus. Each estimator sees *only* the sample and
    is scored (LMSE, Eq. 6.9 — lower is better) against the reference
    distribution.

    Returns a one-row DataFrame with LMSE scores for every estimator,
    keyed by estimator name (matching the Ch.6 result tables).
    """
    eval_df, ref_df = split_corpus(df, eval_frac=eval_frac, seed=seed, text_col=text_col)
    eval_texts = eval_df[text_col].tolist()
    ref_texts = ref_df[text_col].tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Reference (ground-truth) distribution from the reference set.
    ref_dist = mle_estimate(ref_texts, tokenizer=tokenizer, model_name=model_name)

    # Standard estimators from the evaluation (sample) set.
    std = standard_estimators(eval_texts, tokenizer=tokenizer, model_name=model_name)

    # LERF estimate.
    lerf = lerf_estimate(eval_texts, model_name=model_name, device=device)

    row = {"LERF": lmse(lerf, ref_dist), "MLE": lmse(mle_estimate(eval_texts, tokenizer=tokenizer, model_name=model_name), ref_dist)}
    for name, p in std.items():
        row[name] = lmse(p, ref_dist)
    return pd.DataFrame([row])