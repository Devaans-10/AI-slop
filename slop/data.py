"""Dataset utilities.

Every example is a JSON object with these fields:

    text      str   the passage
    label     int   0 = human-written, 1 = AI-generated
    domain    str   e.g. "wikipedia", "reddit", "fiction", "news", "email"
    source    str   the generator, e.g. "human", "gpt-5-nano", "llama-3"
    pair_id   str   id shared by a human text and its AI rewrite

`pair_id` matters: if a human paragraph and its AI rewrite land in different
splits, the model can learn the topic instead of the writing style, and the
validation numbers will look better than they should.
"""

import json
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_by_pair(rows, val_frac=0.1, test_frac=0.1, seed=42):
    """Split so that all rows sharing a pair_id stay in the same split."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["pair_id"]].append(row)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    n = len(keys)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_keys = keys[:n_test]
    val_keys = keys[n_test:n_test + n_val]
    train_keys = keys[n_test + n_val:]

    def collect(ks):
        return [row for k in ks for row in groups[k]]

    return collect(train_keys), collect(val_keys), collect(test_keys)


def validate_rows(rows):
    """Fail early on malformed data rather than halfway through training."""
    required = {"text", "label", "domain", "source", "pair_id"}
    for i, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"row {i} is missing fields: {sorted(missing)}")
        if row["label"] not in (0, 1):
            raise ValueError(f"row {i} has label {row['label']!r}; expected 0 or 1")
        if not row["text"].strip():
            raise ValueError(f"row {i} has empty text")
    return rows
