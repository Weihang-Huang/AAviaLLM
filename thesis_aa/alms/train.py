"""Further pretraining of authorial GPT-2 models (Ch.5 Sec 5.3.1).

==========================================================================
  WHAT THIS MODULE DOES (for newcomers)
==========================================================================

The first stage of the ALMs method is to take a *base* GPT-2 (the same
pretrained model for every author) and "further pretrain" one copy of it on
each candidate author's known writings. The result is one fine-tuned model
per author — an "Authorial Language Model" (ALM). Later, ``alms/ppl.py``
scores a questioned document under each ALM and attributes it to the author
whose model finds it most predictable (lowest perplexity).

**Why "further pretraining" and not "training from scratch"?** GPT-2 is
already fluent in English from its original pretraining. Further pretraining
just nudges its parameters so it's better at predicting *this particular
author's* word sequences. This preserves the model's general fluency while
specialising it — and it needs far less data and compute than pretraining
from scratch (Ch.5 Sec 5.3.1, citing Gururangan et al., 2020).

**Why one model per author (not one shared model)?** The thesis argues
(Ch.5 Sec 5.2) that a single LLM cannot capture the distinct styles of many
authors simultaneously — this is why the earlier "pALM" approach (one model
with per-author output heads) performed poorly. ALMs instead give each
author their own model, so each model's parameters are wholly devoted to
that author's style.

==========================================================================
  THE TRAINING RECIPE (Ch.5 Table 2)
==========================================================================

The thesis uses these hyperparameters (and this module defaults to them):

    Epochs                     100   (use 1-2 for CPU debugging)
    Learning rate              2e-5
    Weight decay               0.01
    Gradient accumulation      64    (effective batch size = 64 x batch_size)
    Block size                 128   (chunk texts into 128-token windows)

We keep the exact recipe but make several things CPU-friendly and
resumable:

  * **fp16 auto-detection** — fp16 is only enabled when CUDA is available.
    The xpu build of torch in this repo doesn't support ``Trainer``'s fp16
    path, so we gate on CUDA specifically. Override with the ``fp16`` argument.

  * **Resumable training** — training 50 authors for 100 epochs each takes
    a long time. If the process crashes halfway, you don't want to redo the
    completed authors. So we write each completed author tag to
    ``log/done.txt`` and skip it on restart. ``log/target.txt`` (one tag per
    line) optionally restricts which authors to train; if absent, all tags
    in the training set are trained.

  * **Configurable block size & batch size** — defaults match the thesis
    but can be lowered (e.g. ``block_size=64, batch_size=1``) so the
    pipeline runs in seconds on CPU with synthetic data.

==========================================================================
  DATA FLOW (step by step)
==========================================================================

For each author tag ``a``:

  1. Filter the training dataframe to rows where ``author_tag == a``.
  2. Split 80/20 into train/eval (the eval split is used only to report a
     held-out perplexity; it does NOT become the test set for attribution).
  3. Tokenise all texts with the GPT-2 tokenizer.
  4. Concatenate all tokenised texts into one long stream and chop it into
     fixed-length ``block_size`` chunks (the standard causal-LM training
     data prep; see the ``group_texts`` function in the HuggingFace docs).
  5. Further-pretrain GPT-2 on those chunks for ``epochs`` epochs.
  6. Save the fine-tuned model + tokenizer to ``models/<author_tag>/``.
  7. Append ``a`` to ``log/done.txt`` so it's skipped on restart.

The saved model directory is what ``alms/ppl.py`` later loads to score
questioned documents.
"""

from __future__ import annotations

import math
import os
import shutil

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .. import config


def _read_log_lines(path: str) -> list[str]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _write_done(log_home: str, tag: str) -> None:
    done_log = os.path.join(log_home, "done.txt")
    os.makedirs(log_home, exist_ok=True)
    with open(done_log, "a", encoding="utf-8") as f:
        f.write(tag + "\n")


def _filter_by_author(
    train_df: pd.DataFrame, author_tag: str, tag_col: str = "author_tag",
) -> Dataset:
    """Filter a dataframe to one author and return a HuggingFace Dataset."""
    sub = train_df[train_df[tag_col] == author_tag].reset_index(drop=True)
    # ``Dataset.from_pandas`` may carry a leftover index column; drop it.
    cols = [c for c in sub.columns if c != "__index_level_0__"]
    return Dataset.from_pandas(sub[cols])


def _preprocess(examples, tokenizer):
    return tokenizer(examples["text"])


def _group_texts(examples, block_size: int = 128):
    """Concatenate and chunk token sequences to ``block_size`` (LMTrain-GPU)."""
    concatenated = {k: sum(v, []) for k, v in examples.items()}
    total_length = len(concatenated[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [tok[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, tok in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def train_authorial_model(
    author_tag: str,
    train_df: pd.DataFrame,
    base_model: str = config.DEFAULT_BASE_MODEL,
    out_dir: str = config.MODEL_DIR,
    tag_col: str = "author_tag",
    epochs: int | None = None,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 64,
    block_size: int = 128,
    batch_size: int = 2,
    fp16: bool | None = None,
    eval_test_size: float = 0.2,
    buffer_dir: str = "gpt2-buffer",
    seed: int = 0,
) -> str:
    """Further-pretrain one GPT-2 on a single author's writings.

    Returns the path to the saved model directory.
    """
    if epochs is None:
        epochs = config.ALMS_TRAIN_CONFIG["epochs"]
    if fp16 is None:
        # Auto-detect: enable fp16 only when a CUDA GPU is available. We do
        # NOT read ``config.ALMS_TRAIN_CONFIG["fp16"]`` here because that flag
        # is a static default (False for CPU debugging) and would otherwise
        # prevent fp16 from ever being enabled on a real GPU run. The xpu
        # build of torch used in this repo does not support ``fp16`` via the
        # ``Trainer`` API, so we gate on CUDA specifically.
        fp16 = torch.cuda.is_available()

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, author_tag)

    # Already trained? Skip. (The marker is the model's config.json —
    # written last-ish by save_model, so its presence means the save
    # completed. NOTE: use isfile, not isdir — this path points *at* the
    # file, not its containing directory.)
    if os.path.isfile(os.path.join(save_path, "config.json")):
        print(f"[ALMs] {author_tag}: already trained, skipping.")
        return save_path

    target_set = _filter_by_author(train_df, author_tag, tag_col)
    if len(target_set) == 0:
        raise ValueError(f"No training rows for author '{author_tag}'.")

    dataset = target_set.train_test_split(test_size=eval_test_size, shuffle=True, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    lm_dataset = dataset.map(
        _preprocess, fn_kwargs={"tokenizer": tokenizer},
        batched=True, num_proc=1,
        remove_columns=dataset["train"].column_names,
    )
    lm_dataset = lm_dataset.map(
        _group_texts, fn_kwargs={"block_size": block_size},
        batched=True, num_proc=1,
    )

    model = AutoModelForCausalLM.from_pretrained(base_model)

    training_args = TrainingArguments(
        output_dir=buffer_dir,
        eval_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=fp16,
        push_to_hub=False,
        report_to=[],
        save_strategy="no",
        logging_steps=50,
        disable_tqdm=False,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    eval_results = trainer.evaluate()
    eval_loss = eval_results["eval_loss"]
    perplexity = math.exp(eval_loss) if eval_loss < 50 else float("inf")
    eval_str = f"Perplexity: {perplexity:.2f}"
    print(f"[ALMs] {author_tag}: {eval_str}")
    with open(os.path.join(save_path, "eval.txt"), "a", encoding="utf-8") as f:
        f.write(eval_str + "\n")

    if os.path.isdir(buffer_dir):
        shutil.rmtree(buffer_dir)

    return save_path


def train_all_authors(
    train_df: pd.DataFrame,
    base_model: str = config.DEFAULT_BASE_MODEL,
    out_dir: str = config.MODEL_DIR,
    tag_col: str = "author_tag",
    log_home: str = config.LOG_DIR,
    **kwargs,
) -> dict[str, str]:
    """Train one authorial model per unique ``author_tag``.

    Resumable: completed authors are listed in ``log_home/done.txt`` and
    skipped on restart. ``log_home/target.txt`` (one tag per line) optionally
    restricts which authors to train; if absent, all tags are trained.
    """
    os.makedirs(log_home, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    target_log = os.path.join(log_home, "target.txt")
    if os.path.isfile(target_log):
        target_lines = _read_log_lines(target_log)
    else:
        target_lines = sorted(train_df[tag_col].unique().tolist())
    done_lines = _read_log_lines(os.path.join(log_home, "done.txt"))
    to_do = [t for t in target_lines if t not in done_lines]

    print(f"[ALMs] Fetched {len(to_do)} tasks: {to_do}")

    results: dict[str, str] = {}
    for tag in to_do:
        save_path = train_authorial_model(
            tag, train_df, base_model=base_model, out_dir=out_dir,
            tag_col=tag_col, **kwargs,
        )
        _write_done(log_home, tag)
        results[tag] = save_path
    return results