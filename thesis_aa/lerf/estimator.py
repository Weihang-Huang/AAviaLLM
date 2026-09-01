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
  GOODNESS-OF-FIT: LMSE (Ch.6 Sec 6.5.4)
==========================================================================

LMSE = Log Mean Squared Error = mean of ``(log p_est - log p_ref)^2`` over
all vocabulary types. We compare in *log* space because relative frequencies
span many orders of magnitude. The thesis reports LMSE as *negative* (so
lower/more-negative = better fit); ``lmse`` returns the negated value to
match that convention. Identical distributions give LMSE = 0.

==========================================================================
  HOW TO USE (minimal example)
==========================================================================

    from thesis_aa import data as data_mod
    from thesis_aa.lerf import estimator as lerf_est

    df, _ = data_mod.generate_synthetic()
    row = lerf_est.evaluate_all_estimators(df, model_name='gpt2')
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

def lerf_estimate(
    texts: list[str] | pd.Series,
    model_name: str = "gpt2",
    model=None,
    tokenizer=None,
    device: torch.device | None = None,
    stride: int = 1024,
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

    for text in tqdm(texts, desc="LERF"):
        clean = str(text).replace("<BOS>", "").replace("<EOS>", "")
        if not clean.strip():
            continue
        enc = tokenizer(clean, return_tensors="pt")
        ids = enc.input_ids[0]
        seq_len = ids.size(0)
        if seq_len < 2:
            continue

        # Sliding window to handle long texts; the original uses a stride
        # equal to the model's context. We accumulate the softmax over the
        # *predicted* token at each position (logits at position i predict i+1).
        max_len = model.config.n_positions
        for begin in range(0, seq_len - 1, stride):
            end = min(begin + max_len, seq_len)
            chunk = ids[begin:end].unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(chunk).logits  # (1, L, V)
                probs = F.softmax(logits, dim=-1)  # (1, L, V)
            # Position i predicts token i+1. Sum over positions 0..L-2.
            acc += probs[0, :-1, :].sum(dim=0).double()

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
    """Katz backoff approximation using Good-Turing discounting (Ch.6 Sec 6.3.1).

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

    discounted = np.zeros_like(counts, dtype=float)
    for r in range(1, len(freq_of_freq)):
        if freq_of_freq[r] > 0:
            discounted[counts == r] = counts[counts == r] * gt_factor[r]

    freed_mass = (counts * (1 - np.where(counts > 0, gt_factor[counts.astype(int)], 1.0))).sum()
    n_unseen = (counts == 0).sum()

    p = discounted / N if N > 0 else discounted
    if n_unseen > 0 and freed_mass > 0:
        p[counts == 0] = freed_mass / n_unseen / N
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

def lmse(estimated: np.ndarray, reference: np.ndarray, eps: float = 1e-12) -> float:
    """Log Mean Squared Error between estimated and reference distributions.

    LMSE = mean( (log p_est - log p_ref)^2 ), summed in log space. Following
    the thesis (Ch.6 Table), lower (more negative) values indicate better fit,
    so we return the negative mean squared log-error for consistency with the
    reported tables.
    """
    est = np.asarray(estimated, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    mask = (est > 0) | (ref > 0)
    log_est = np.log(est[mask] + eps)
    log_ref = np.log(ref[mask] + eps)
    mse = np.mean((log_est - log_ref) ** 2)
    return -float(mse)  # more negative = better (per thesis tables)


# ---------------------------------------------------------------------------
# Corpus splitting (Ch.6 Sec 6.5.2)
# ---------------------------------------------------------------------------

def split_corpus(df: pd.DataFrame, eval_frac: float = 0.2, seed: int = 0,
                 text_col: str = "text", tag_col: str = "author_tag") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a corpus into a small evaluation (sample) set and a reference set.

    The evaluation set acts as the sample corpus; the reference set represents
    the full population whose distribution we wish to estimate (Ch.6 Sec
    6.5.2). The intuition is a lab experiment: you get to *see* only a small
    sample (the eval set) and must estimate the word-frequency table of the
    whole variety; the reference set is the "answer key" you score against.

    The shuffle is a plain row-level cut (NOT per-author stratified) because
    the LERF question is about the corpus as one variety of language, not
    about per-author balance.

    Returns ``(eval_df, reference_df)``.
    """
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cut = int(len(shuffled) * eval_frac)
    return shuffled.iloc[:cut], shuffled.iloc[cut:]


# ---------------------------------------------------------------------------
# Full estimator evaluation (Ch.6 Sec 6.5)
# ---------------------------------------------------------------------------

def evaluate_all_estimators(
    df: pd.DataFrame,
    model_name: str = "gpt2",
    eval_frac: float = 0.2,
    seed: int = 0,
    device: torch.device | None = None,
    text_col: str = "text",
) -> pd.DataFrame:
    """Evaluate LERF + all standard estimators on one corpus split.

    Returns a one-row DataFrame with LMSE scores for every estimator, keyed
    by estimator name (matching the Ch.6 result tables).
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