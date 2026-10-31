"""Score text with a trained adapter.

Usage:
    python -m slop.predict --adapter checkpoints/run1/adapter --file essay.txt
    echo "Some text" | python -m slop.predict --adapter checkpoints/run1/adapter

Output per text:
    score     mean probability that the text is AI-generated, over all windows
    max       highest window score (useful for spotting one AI-written section)
    label     "ai" if score >= THRESHOLD, else "human"
    paragraphs  per-paragraph scores, so you can see which parts drove the result
"""

import argparse
import json
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from slop.model import BASE_MODEL, window_encode

# Chosen on the validation split, not on test. Revisit it after retraining.
THRESHOLD = 0.5


def load(adapter_dir, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.to(device).eval()
    return model, tokenizer


@torch.no_grad()
def window_probs(model, tokenizer, texts, device, batch_size=32):
    """Return (per-text lists of window probabilities)."""
    enc = window_encode(tokenizer, texts)
    owner = enc.pop("owner")
    n = len(enc["input_ids"])
    probs = []
    for start in range(0, n, batch_size):
        chunk = {k: v[start:start + batch_size] for k, v in enc.items()}
        padded = tokenizer.pad(chunk, return_tensors="pt").to(device)
        logits = model(**padded).logits
        probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())

    grouped = [[] for _ in texts]
    for o, p in zip(owner, probs):
        grouped[o].append(p)
    return grouped


def score_texts(model, tokenizer, texts, device=None):
    """Mean AI probability per text. Used by training for per-group reports."""
    device = device or next(model.parameters()).device
    grouped = window_probs(model, tokenizer, texts, device)
    return [sum(g) / len(g) if g else 0.0 for g in grouped]


def split_paragraphs(text):
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def analyze(model, tokenizer, text, device=None):
    device = device or next(model.parameters()).device
    paragraphs = split_paragraphs(text) or [text]

    doc_windows = window_probs(model, tokenizer, [text], device)[0]
    para_windows = window_probs(model, tokenizer, paragraphs, device)
    para_scores = [sum(w) / len(w) for w in para_windows]

    score = sum(doc_windows) / len(doc_windows)
    return {
        "score": round(score, 4),
        "max": round(max(doc_windows), 4),
        "label": "ai" if score >= THRESHOLD else "human",
        "n_windows": len(doc_windows),
        "paragraphs": [
            {"text": p[:120], "score": round(s, 4)}
            for p, s in zip(paragraphs, para_scores)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--file", help="read text from this file instead of stdin")
    args = parser.parse_args()

    text = open(args.file, encoding="utf-8").read() if args.file else sys.stdin.read()
    if not text.strip():
        sys.exit("no text provided")

    model, tokenizer = load(args.adapter)
    print(json.dumps(analyze(model, tokenizer, text), indent=2))


if __name__ == "__main__":
    main()
