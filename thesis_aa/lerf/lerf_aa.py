"""LERF-AA: LERF features for supervised authorship attribution (Ch.7).

==========================================================================
  WHAT THIS MODULE DOES (for newcomers)
==========================================================================

LERF-AA is the third and final method in the thesis. It takes the LERF
relative-frequency estimator from Ch.6 (which uses a *frozen* GPT-2 — no
fine-tuning) and uses it as a *feature extractor* for ordinary supervised
classifiers. The idea: instead of classifying documents by hand-crafted
stylometric features (function-word counts, n-grams, etc.), represent each
document by its LERF profile — a 50,257-dim vector of LLM-estimated relative
frequencies — and let a standard classifier learn which authors that
profile distinguishes.

**How LERF-AA differs from ALMs (Ch.5):**
  - ALMs fine-tunes a *separate* GPT-2 per author and attributes by perplexity.
  - LERF-AA uses *one shared* frozen GPT-2 for every document and attributes
    with a supervised classifier on top of the LERF feature vectors.
  - ALMs models "realised language" (the actual tokens in the questioned
    document); LERF-AA models "expected language" (the vocabulary-wide
    distribution the LLM expects given the document's contexts). The thesis
    (Ch.8 Sec 8.1.1) frames this as the core conceptual contrast.

==========================================================================
  THE THREE-STEP PIPELINE (Ch.7 Sec 7.3)
==========================================================================

Step 1 — Feature extraction (``extract_lerf_features``):
    Each document is treated as an independent "sample corpus" and fed
    through ``lerf.estimator.lerf_estimate``. The result is a
    ``(n_documents, 50257)`` matrix — one LERF profile per document. The
    model is *frozen* (no training): every document gets the same GPT-2.

Step 2 — Feature selection (``select_mfw``):
    The full 50,257-dim vector is often more than a classifier needs. We
    rank vocabulary types by their *observed training* frequency and keep
    only the top-``k`` (k in {50, 100, 150, 200, 500, 1000, full}). The
    ranking uses *training* data only, so test documents never influence
    which features are selected (this is standard MFW selection; Ch.7
    Sec 7.3.2).

Step 3 — Classification (``build_classifiers`` + ``run_lerf_aa``):
    Eight classifiers are trained on the (selected) feature vectors and
    author labels. Each classifier predicts the author of each test
    document and produces a ranked candidate list (for top-N metrics).

==========================================================================
  THE EIGHT CLASSIFIERS (Ch.7 Sec 7.3.3, exact configs)
==========================================================================

The thesis deliberately spans classifier families (margin / probabilistic /
tree ensemble / boosting) to separate the representation's discriminative
power from any one classifier's inductive bias. Configurations are taken
verbatim from the manuscript:

  - **Linear SVM**        SGDClassifier(loss="hinge") — the classic
                          stylometric workhorse; O(n) scaling suits the
                          50k-dim feature space.
  - **Logistic Reg**      SGDClassifier(loss="log_loss") — same linear family
                          as SVM but calibrated probabilities.
  - **Random Forest**     100 trees, max_features="sqrt" — nonlinear,
                          handles the wide value range of LERF features.
  - **Extra Trees**       100 trees, max_depth=50, max_features="sqrt" —
                          more randomised than RF; tests whether random
                          thresholds help.
  - **Gaussian NB**       default — fast generative baseline; well-motivated
                          because LERF estimates are interpretable as
                          probabilities (bag-of-words assumption).
  - **Decision Tree**     max_depth=50, max_features="sqrt" — interpretable
                          single tree; an ablation reference for the ensembles.
  - **AdaBoost**          30 stumps (depth-1) — boosting over very shallow
                          base learners; tests iterative reweighting.
  - **Hist GB**           30 iters, max_depth=3 — histogram-based gradient
                          boosting (the classical GBM is infeasible at 50k
                          dims because of its O(n*d) per-split cost).

**Why these eight and not others?** The thesis excludes classifiers whose
complexity scales super-linearly with feature dimension (e.g. libsvm SVC,
k-NN, deep MLPs) because at 50,257 features they are either memory-bound or
overfit severely given the modest number of training documents per author.

==========================================================================
  HOW TO USE (minimal example)
==========================================================================

    from thesis_aa import config, data as data_mod
    from thesis_aa.lerf import lerf_aa

    train_df, test_df = data_mod.generate_synthetic()
    results = lerf_aa.full_pipeline(
        train_df, test_df,
        model_name='gpt2',           # or 'gpt2-xl' for the thesis config
        mfw_sizes=[50, 100],         # or config.MFW_SIZES for all seven
    )
    for k, df in results.items():
        print(f'--- MFW={k} ---')
        print(df[['classifier', 'macro_accuracy', 'top_1', 'top_5']])
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from .. import config, eval as eval_mod
from .estimator import count_token_ids, lerf_estimate

VOCAB_SIZE = config.GPT2_VOCAB_SIZE


# ---------------------------------------------------------------------------
# Step 1: per-document LERF feature extraction (Sec 7.3.1)
# ---------------------------------------------------------------------------

def extract_lerf_features(
    df: pd.DataFrame,
    model_name: str = "gpt2",
    device: torch.device | None = None,
    text_col: str = "text",
    show_progress: bool = True,
) -> np.ndarray:
    """Extract a LERF profile for every document.

    Returns a ``(n_documents, vocab_size)`` matrix. Each row is the LERF
    estimate obtained by treating that single document as the sample corpus
    (Ch.7 Sec 7.3.1). The model is parameter-frozen and shared across all
    documents — no per-author fine-tuning (contrast with ALMs).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = config.get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    n = len(df)
    X = np.zeros((n, model.config.vocab_size), dtype=np.float64)
    it = tqdm(range(n), desc="LERF-AA features") if show_progress else range(n)
    for i in it:
        X[i] = lerf_estimate(
            [df[text_col].iloc[i]], model=model, tokenizer=tokenizer, device=device,
        )
    return X


# ---------------------------------------------------------------------------
# Step 2: most-frequent-word selection (Sec 7.3.2)
# ---------------------------------------------------------------------------

def select_mfw(
    X_train: np.ndarray,
    X_test: np.ndarray,
    train_texts: list[str] | pd.Series,
    k: int,
    tokenizer=None,
    model_name: str = "gpt2",
    vocab_size: int = VOCAB_SIZE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank vocabulary by observed *training* frequency and retain top-k.

    **What MFW selection does and why:** The full LERF vector has 50,257
    dimensions — far more than many classifiers need, and much of it is
    near-zero for rare/unseen types. "Most-Frequent-Word" selection keeps only
    the ``k`` vocabulary types that occur most often in the *training* data,
    reducing dimensionality and often improving accuracy (Ch.7 Sec 7.3.2).

    **Why training-only ranking?** The ranking is derived exclusively from
    the training partition and then *applied unchanged* to the test data. If
    we ranked using the test data we would be leaking test information into
    feature selection — a subtle form of data leakage that inflates
    accuracy. The ``indices`` returned are the column positions of the
    selected types; the same indices select columns from both ``X_train``
    and ``X_test``.

    Returns ``(X_train_sel, X_test_sel, indices)``. ``k`` may be
    ``>= vocab_size`` to keep the full vocabulary (handled at the top of the
    function).
    """
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if k >= vocab_size:
        return X_train, X_test, np.arange(vocab_size)

    # Observed training frequency (shared counting helper from the
    # estimator module — previously a duplicated tokenize-and-tally loop).
    counts = count_token_ids(
        train_texts, tokenizer=tokenizer, vocab_size=vocab_size,
    )
    top_k = np.argsort(counts)[::-1][:k]
    return X_train[:, top_k], X_test[:, top_k], top_k


# ---------------------------------------------------------------------------
# Step 3: eight classifiers (Sec 7.3.3, exact configurations)
# ---------------------------------------------------------------------------

def build_classifiers() -> Dict[str, object]:
    """Instantiate the eight classifiers from Ch.7 Sec 7.3.3.

    Configurations are taken verbatim from the manuscript:
      - Linear SVM       : SGDClassifier(loss="hinge")
      - Logistic Reg     : SGDClassifier(loss="log_loss")
      - Random Forest    : 100 trees, max_features="sqrt"
      - Extra Trees      : 100 trees, max_depth=50, max_features="sqrt"
      - Gaussian NB      : default
      - Decision Tree    : max_depth=50, max_features="sqrt"
      - AdaBoost         : 30 stumps (depth-1), max_features="sqrt" via base estimator
      - Hist GB          : 30 iters, max_depth=3

    **Reproducibility note.** Six of the eight are stochastic (SGD shuffling,
    random feature subsets, bootstrap sampling). Without a seed, repeated
    runs on identical data produce different accuracy tables — which makes
    debugging and regression-testing impossible. We pin ``random_state=0``
    everywhere it is accepted; the configurations are otherwise verbatim.
    """
    clf = {
        "Linear SVM": SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3,
                                    random_state=0, n_jobs=-1),
        "Logistic Regression": SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3,
                                             random_state=0, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_features="sqrt",
                                                random_state=0, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=50,
                                           max_features="sqrt", random_state=0, n_jobs=-1),
        "Gaussian Naive Bayes": GaussianNB(),  # deterministic (closed-form)
        "Decision Tree": DecisionTreeClassifier(max_depth=50, max_features="sqrt",
                                                 random_state=0),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=30,
            estimator=DecisionTreeClassifier(max_depth=1, random_state=0),
            random_state=0,
        ),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=30, max_depth=3, random_state=0,
        ),
    }
    return clf


# ---------------------------------------------------------------------------
# Run one (classifier, feature-set) configuration
# ---------------------------------------------------------------------------

def run_lerf_aa(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifiers: Dict[str, object] | None = None,
) -> pd.DataFrame:
    """Fit each classifier, predict, and compute evaluation metrics.

    Returns a DataFrame with one row per classifier: macro-accuracy, top-N
    accuracies (N=1..5), and true-author-rank stats. Decision-function values
    are used for ranking where available, else predicted class probabilities.
    """
    if classifiers is None:
        classifiers = build_classifiers()
    classes = list(np.unique(y_train))
    rows = []

    for name, clf in classifiers.items():
        try:
            clf_clone = type(clf)(**clf.get_params())
        except TypeError:
            import copy
            clf_clone = copy.deepcopy(clf)
        clf_clone.fit(X_train, y_train)

        pred = clf_clone.predict(X_test)
        # Build rankings per test instance (batched: one call for ALL test
        # documents instead of one sklearn call per document — predict_proba
        # on tree ensembles is the expensive part on 50k-dim inputs).
        scores_mat = _batched_scores(clf_clone, X_test, classes)
        rank_list = []
        for i in range(len(X_test)):
            ordered = [c for c, _ in sorted(zip(classes, scores_mat[i]), key=lambda x: -x[1])]
            rank_list.append(ordered)

        row = {
            "classifier": name,
            "macro_accuracy": eval_mod.macro_accuracy(y_test, pred),
        }
        for n in range(1, 6):
            row[f"top_{n}"] = eval_mod.top_n_accuracy(rank_list, y_test, n)
        rank_stats = eval_mod.true_author_rank_stats(rank_list, y_test)
        row.update({f"rank_{k}": v for k, v in rank_stats.items()})
        rows.append(row)

    return pd.DataFrame(rows)


def _batched_scores(clf, X: np.ndarray, classes) -> np.ndarray:
    """Return an ``(n_test_docs, n_classes)`` per-class score matrix.

    Ranking candidates requires a per-class score. Different classifier
    families expose this differently:

      * **Margin classifiers** (SVM, LogReg via SGD) provide
        ``decision_function`` — the signed distance to the decision
        boundary. Higher = more confident the instance belongs to that
        class. We sort descending so the best class is first.

      * **Probabilistic classifiers** (Naive Bayes, tree ensembles) provide
        ``predict_proba`` — a probability per class.

      * **Fallback**: if neither is available, we one-hot encode the
        ``predict`` results (rank degenerates to putting the predicted
        class first, all others tied).

    **Binary edge case:** For a 2-class problem ``decision_function`` returns
    a 1-D array (the score for the *positive* class only). We expand it to
    ``[s, -s]`` per instance so the two classes get symmetric scores and the
    ranking still works. This case doesn't arise in the thesis (>=7
    candidate authors) but we handle it for correctness on tiny tests.

    **Batching note:** this replaces the former one-instance-at-a-time
    helper (``_per_instance_scores``); computing scores for all test
    documents in one call avoids per-call overhead and lets tree ensembles
    share work across rows. The score *values* are identical.
    """
    # Margin scores first (SVM/LogReg families).
    try:
        S = clf.decision_function(X)
        S = np.asarray(S)
        if S.ndim == 1:  # binary: (n,) -> (n, 2) with symmetric scores
            S = np.column_stack([S, -S])
        if S.shape[0] == len(X):
            return S
    except Exception:
        pass
    # Probabilities (NB / tree ensembles / boosting families).
    try:
        P = np.asarray(clf.predict_proba(X))
        if P.shape[0] == len(X):
            return P
    except Exception:
        pass
    # Fallback: one-hot of the predicted classes.
    preds = np.asarray(clf.predict(X))
    return np.array([[1.0 if c == p else 0.0 for c in classes] for p in preds])


# ---------------------------------------------------------------------------
# Full pipeline (Sec 7.4)
# ---------------------------------------------------------------------------

def full_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "gpt2",
    device: torch.device | None = None,
    mfw_sizes: list[int] | None = None,
    classifiers: Dict[str, object] | None = None,
    tag_col: str = "author_tag",
    text_col: str = "text",
    out_dir: str | None = None,
) -> Dict[int, pd.DataFrame]:
    """Run the full LERF-AA evaluation across feature-set sizes and classifiers.

    Extracts per-document LERF features once, then iterates over the MFW sizes
    in ``mfw_sizes`` (default: the seven configurations from Sec 7.3.2).
    Returns a dict mapping each ``k`` to a results DataFrame.
    """
    if mfw_sizes is None:
        mfw_sizes = list(config.MFW_SIZES)
    if out_dir is None:
        out_dir = config.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    X_train = extract_lerf_features(train_df, model_name=model_name, device=device, text_col=text_col)
    X_test = extract_lerf_features(test_df, model_name=model_name, device=device, text_col=text_col)
    y_train = train_df[tag_col].to_numpy()
    y_test = test_df[tag_col].to_numpy()

    results: Dict[int, pd.DataFrame] = {}
    for k in mfw_sizes:
        Xtr, Xte, _ = select_mfw(X_train, X_test, train_df[text_col].tolist(), k)
        df_res = run_lerf_aa(Xtr, y_train, Xte, y_test, classifiers=classifiers)
        df_res["mfw_size"] = k
        out_path = os.path.join(out_dir, f"lerf_aa_mfw-{k}.csv")
        df_res.to_csv(out_path, index=False)
        print(f"[LERF-AA] MFW={k} -> {out_path}")
        results[k] = df_res
    return results