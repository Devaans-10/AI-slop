"""Rewrite human-written essays with Gemini and emit matching AI rows.

Usage:
    python -m slop.generate_ai_essays --input human_essays.jsonl \
        --output data/processed/ai_essays.jsonl

Reads one JSON object per line with fields ``text``, ``domain`` and
``pair_id`` and writes one row per input row:

    {"text": "...", "label": 1, "domain": "essay", "source": "gemini", "pair_id": "..."}

Existing output rows are skipped by ``pair_id`` so the script can resume after
a crash, and output is appended one line at a time as each row finishes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from slop.data import read_jsonl

REWRITE_PROMPT = (
    "Rewrite the essay below so it keeps the same meaning, facts, and "
    "approximate length, but in fluent, polished prose. Do not add new claims, "
    "do not summarize, and do not change the topic. Output only the rewritten "
    "essay, with no quotation marks, no preamble, and no explanation.\n\n"
    "Essay:\n{text}"
)

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_DELAY = 1.0
MAX_TRIES = 5
SOURCE = "gemini"
LABEL_AI = 1

# Status codes worth retrying: rate limit and server errors.
RETRYABLE_CODES = {429, 500, 502, 503, 504}


def _retryable(exc: Exception) -> bool:
    """Return True if the exception looks like a retriable API error."""
    code = getattr(exc, "code", None)
    if code is None:
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
    return code in RETRYABLE_CODES


def load_done_pair_ids(path: Path) -> set[str]:
    """Return the pair_ids already present in the output file, for resuming."""
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pair_id = row.get("pair_id")
            if pair_id is not None:
                done.add(pair_id)
    return done


def rewrite_essay(client: object, model: str, text: str) -> str:
    """Rewrite one essay with retries for rate limit and server errors."""
    prompt = REWRITE_PROMPT.replace("{text}", text)
    last_exc: Exception | None = None
    for attempt in range(1, MAX_TRIES + 1):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return (response.text or "").strip()
        except Exception as exc:  # noqa: BLE001 - retry or fail
            last_exc = exc
            if not _retryable(exc) or attempt == MAX_TRIES:
                break
            time.sleep(min(2 ** (attempt - 1), 30))
    assert last_exc is not None
    raise last_exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite human essays with Gemini into matching AI rows."
    )
    parser.add_argument("--input", required=True, help="input JSONL of human essays")
    parser.add_argument("--output", required=True, help="output JSONL of AI rows")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name")
    parser.add_argument(
        "--limit", type=int, default=None, help="process only the first N rows"
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY, help="seconds between requests"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the first row's prompt and exit without calling the API",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        rows = read_jsonl(args.input)
        if args.limit is not None:
            rows = rows[: args.limit]
        if rows:
            print(REWRITE_PROMPT.replace("{text}", rows[0]["text"]))
        else:
            print("no rows in input")
        return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("error: GEMINI_API_KEY environment variable is not set", file=sys.stderr)
        sys.exit(1)

    from google import genai

    client = genai.Client(api_key=api_key)

    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = read_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    done = load_done_pair_ids(output_path)
    pending = [row for row in rows if row.get("pair_id") not in done]
    skipped = len(rows) - len(pending)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_empty = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for i, row in enumerate(pending, start=1):
            text = rewrite_essay(client, args.model, row["text"])
            if not text:
                n_empty += 1
                continue
            out_row = {
                "text": text,
                "label": LABEL_AI,
                "domain": row.get("domain", "essay"),
                "source": SOURCE,
                "pair_id": row.get("pair_id"),
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            f.flush()
            n_written += 1
            if i % 10 == 0:
                print(
                    f"processed {i}/{len(pending)} rows "
                    f"(written={n_written}, empty_skipped={n_empty})"
                )
            time.sleep(args.delay)

    print(
        f"done: {n_written} written, {n_empty} empty outputs skipped, "
        f"{skipped} already in output"
    )


if __name__ == "__main__":
    main()
