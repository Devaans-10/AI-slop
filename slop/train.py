"""Fine-tune a LoRA adapter on bert-base-cased for AI-text detection.

Usage (Colab GPU or local):
    python -m slop.train --train data/processed/train.jsonl \
        --val data/processed/val.jsonl --test data/processed/test.jsonl \
        --out checkpoints/run1 --epochs 3

Add --extra for additional JSONL files in the same schema (other domains).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from slop.data import read_jsonl, validate_rows
from slop.model import BASE_MODEL, ID2LABEL, LABEL2ID, window_encode


def load_split(paths):
    rows = []
    for p in paths:
        rows.extend(validate_rows(read_jsonl(p)))
    return rows


def to_dataset(rows, tokenizer):
    """Window every text, then build a HF Dataset of windows."""
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]
    enc = window_encode(tokenizer, texts, labels)
    enc.pop("owner")
    return Dataset.from_dict(enc)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "auc": roc_auc_score(labels, probs),
    }


def per_domain_report(model, tokenizer, rows, device):
    """Accuracy broken down by domain and generator.

    The aggregate score hides the main weakness of this kind of detector,
    so the report is the number to watch when deciding whether a model is
    actually good.
    """
    from slop.predict import score_texts

    texts = [r["text"] for r in rows]
    probs = score_texts(model, tokenizer, texts, device)
    buckets = defaultdict(lambda: [0, 0])  # key -> [correct, total]
    for r, p in zip(rows, probs):
        pred = int(p >= 0.5)
        for key in (f"domain={r['domain']}", f"source={r['source']}"):
            buckets[key][0] += int(pred == r["label"])
            buckets[key][1] += 1
    return {k: {"accuracy": c / t, "n": t} for k, (c, t) in sorted(buckets.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--val", nargs="+", required=True)
    parser.add_argument("--test", nargs="+", default=None)
    parser.add_argument("--extra", nargs="*", default=[],
                        help="extra JSONL files, added to the training set")
    parser.add_argument("--out", default="checkpoints/run1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    train_rows = load_split(args.train + args.extra)
    val_rows = load_split(args.val)
    print(f"train texts={len(train_rows)} val texts={len(val_rows)}")

    train_ds = to_dataset(train_rows, tokenizer)
    val_ds = to_dataset(val_rows, tokenizer)
    print(f"train windows={len(train_ds)} val windows={len(val_ds)}")

    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.1,
        target_modules=["query", "value"],
    )
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()

    out = Path(args.out)
    training_args = TrainingArguments(
        output_dir=str(out / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        logging_steps=50,
        fp16=(device == "cuda"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Save only the adapter (a few MB) plus the tokenizer. The base model is
    # downloaded from Hugging Face at load time.
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")

    results = {"val": trainer.evaluate()}
    if args.test:
        test_rows = load_split(args.test)
        model.eval()
        model.to(device)
        results["test_by_group"] = per_domain_report(model, tokenizer, test_rows, device)
        results["test_overall"] = {
            k.replace("eval_", ""): v
            for k, v in trainer.evaluate(to_dataset(test_rows, tokenizer)).items()
        }

    (out / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
