"""Turn the public Wikipedia human/AI pairs into the project's JSONL schema.

Usage:
    python -m slop.prepare_data --out data/processed

The source dataset is gouwsxander/wikipedia-human-ai on Hugging Face.
Check the column names with `print(ds)` before the first run and adjust
COLUMN_MAP if they differ.

To add your own data (other domains, other generators), write rows in the
same schema with slop.data.write_jsonl and place them alongside the output,
then pass them to train.py with --extra. More domains is the single biggest
improvement you can make, since the original model fails on Reddit,
fiction, and recipes.
"""

import argparse
from pathlib import Path

from slop.data import split_by_pair, validate_rows, write_jsonl

DATASET = "gouwsxander/wikipedia-human-ai"

# Map the source dataset's column names to ours. Adjust after inspecting.
COLUMN_MAP = {
    "human": "human_text",
    "ai": "ai_text",
}


def build_rows(ds):
    rows = []
    for i, ex in enumerate(ds):
        pair_id = f"wiki-{i}"
        rows.append({
            "text": ex[COLUMN_MAP["human"]],
            "label": 0,
            "domain": "wikipedia",
            "source": "human",
            "pair_id": pair_id,
        })
        rows.append({
            "text": ex[COLUMN_MAP["ai"]],
            "label": 1,
            "domain": "wikipedia",
            "source": "gpt-5-nano",
            "pair_id": pair_id,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local", default=None,
                        help="use a local copy of the dataset (JSONL) instead of downloading")
    args = parser.parse_args()

    if args.local:
        from slop.data import read_jsonl
        ds = read_jsonl(args.local)
    else:
        from datasets import load_dataset
        ds = load_dataset(DATASET, split="train")
    print(len(ds), "source rows")

    rows = validate_rows(build_rows(ds))
    train, val, test = split_by_pair(rows, seed=args.seed)

    out = Path(args.out)
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(val, out / "val.jsonl")
    write_jsonl(test, out / "test.jsonl")
    print(f"train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
