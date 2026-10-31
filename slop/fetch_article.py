"""Fetch the main text of a web article through the TinyFish Fetch API.

Usage:
    python -m slop.fetch_article https://example.com/article

Requires the TINYFISH_API_KEY environment variable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any

import requests

TINYFISH_ENDPOINT = "https://api.fetch.tinyfish.ai"
MIN_WORDS = 50
RETRY_WAIT_SECONDS = 2.0
MAX_ATTEMPTS = 2
REQUEST_TIMEOUT_SECONDS = 60
MAX_PARAGRAPHS_DEFAULT = 200

STOP_HEADINGS = {
    "references",
    "external links",
    "see also",
    "notes",
    "further reading",
    "bibliography",
    "software",
    "journals",
    "conferences",
    "books",
    "sources",
}

INCLUDE_SELECTORS = ["#mw-content-text", "article", "main"]
EXCLUDE_SELECTORS = [
    ".comments",
    "nav",
    "footer",
    "header",
    ".newsletter-signup",
    ".related-posts",
    "aside",
    ".mw-references-wrap",
    ".reflist",
    ".navbox",
    ".infobox",
    ".toc",
    ".catlinks",
    ".printfooter",
    ".mw-editsection",
    "table",
    ".hatnote",
    ".sidebar",
    ".metadata",
]


class FetchError(Exception):
    """Raised when an article cannot be fetched or is too short."""


def _api_key() -> str:
    """Return the TinyFish API key, exiting with a clear error if missing."""
    key = os.environ.get("TINYFISH_API_KEY")
    if not key:
        print(
            "error: TINYFISH_API_KEY environment variable is not set",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def validate_url(url: str) -> None:
    """Validate that *url* is an http/https URL.

    Raises:
        ValueError: if the URL does not start with http:// or https://.
    """
    if not isinstance(url, str) or not (
        url.startswith("http://") or url.startswith("https://")
    ):
        raise ValueError(
            f"invalid URL: {url!r}; must start with http:// or https://"
        )


def remove_markdown_links(text: str) -> str:
    """Remove markdown links, keeping only the visible text.

    "[text](url)" becomes "text".
    """
    return re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)


def remove_markdown_markers(text: str) -> str:
    """Remove markdown heading and emphasis markers."""
    text = text.replace("**", "")
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = text.replace("*", "")
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    """Split markdown text into non-empty paragraphs on blank lines.

    Markdown symbols like # and ** are removed from each paragraph.
    """
    return [
        cleaned
        for raw in text.split("\n\n")
        if (cleaned := remove_markdown_markers(raw))
    ]


def cleanup_paragraphs(text: str, max_paragraphs: int = MAX_PARAGRAPHS_DEFAULT) -> list[str]:
    """Clean and split article text into body paragraphs.

    Splits on blank lines, then splits each block on single newlines and keeps
    only lines with at least 12 words ending in sentence punctuation. Stops at
    reference/list headings, drops pipe tables and "Retrieved from" /
    "Categories:" lines, and drops short paragraphs.
    """
    result: list[str] = []
    for raw in text.split("\n\n"):
        cleaned = remove_markdown_links(raw)
        cleaned = remove_markdown_markers(cleaned)
        if not cleaned:
            continue
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        if any(line.lower() in STOP_HEADINGS for line in lines):
            break
        kept_lines = []
        for line in lines:
            if "|" in line:
                continue
            if line.startswith("Retrieved from") or line.startswith("Categories:"):
                continue
            words = len(line.split())
            ends_with_punct = line[-1] in {".", "!", "?", '"', "'"}
            if words >= 12 and ends_with_punct:
                kept_lines.append(line)
        if kept_lines:
            paragraph = " ".join(kept_lines)
            result.append(paragraph)
            if len(result) >= max_paragraphs:
                break
    return result


def word_count(paragraphs: list[str]) -> int:
    """Return the total number of words across all paragraphs."""
    return sum(len(p.split()) for p in paragraphs)


def fetch_article_text(
    url: str, max_paragraphs: int = MAX_PARAGRAPHS_DEFAULT
) -> dict[str, Any]:
    """Fetch and extract the main article text from *url* via TinyFish.

    Returns a dict with keys ``url``, ``title``, ``paragraphs`` and
    ``word_count``.

    The first attempt uses ``include_selectors``. If TinyFish replies with
    ``selector_not_matched``, a single fallback attempt is made without
    ``include_selectors`` (keeping ``exclude_selectors``). Cleanup is applied
    to both results, and a fallback result under ``MIN_WORDS`` words raises a
    ``FetchError`` suggesting a direct article link.

    Raises:
        ValueError: if *url* is not an http/https URL.
        FetchError: if the API reports an error for the URL, or the extracted
            text has fewer than 50 words.
    """
    validate_url(url)

    headers = {
        "X-API-Key": _api_key(),
        "Content-Type": "application/json",
    }
    body = {
        "urls": [url],
        "format": "markdown",
        "links": False,
        "image_links": False,
        "ttl": 0,
        "per_url_timeout_ms": 45000,
        "include_selectors": INCLUDE_SELECTORS,
        "exclude_selectors": EXCLUDE_SELECTORS,
    }

    tried_fallback = False
    last_error: str | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.post(
                TINYFISH_ENDPOINT,
                headers=headers,
                json=body,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.exceptions.Timeout as exc:
            last_error = f"request timed out: {exc}"
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_WAIT_SECONDS)
                continue
            raise FetchError(
                f"request timed out after {MAX_ATTEMPTS} attempt(s): {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise FetchError(f"request failed: {exc}") from exc

        if 500 <= response.status_code < 600:
            last_error = f"TinyFish server error HTTP {response.status_code}"
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_WAIT_SECONDS)
                continue
            raise FetchError(
                f"{last_error} after {MAX_ATTEMPTS} attempt(s)"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise FetchError(f"invalid JSON response: {exc}") from exc

        errors = data.get("errors") or []
        url_error = next((e for e in errors if e.get("url") == url), None)
        if url_error:
            code = url_error.get("error", "unknown")
            status = url_error.get("status")
            detail = f" (HTTP {status})" if status else ""
            last_error = f"TinyFish error for {url}: {code}{detail}"
            if code == "timeout" and attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_WAIT_SECONDS)
                continue
            if (
                code == "selector_not_matched"
                and not tried_fallback
                and attempt < MAX_ATTEMPTS
            ):
                tried_fallback = True
                body = {
                    key: value
                    for key, value in body.items()
                    if key != "include_selectors"
                }
                continue
            raise FetchError(last_error)

        results = data.get("results") or []
        result = next((r for r in results if r.get("url") == url), None)
        if result is None:
            raise FetchError(f"no result returned for {url}")

        title = result.get("title") or ""
        text = result.get("text") or ""
        if not text:
            raise FetchError(f"no extractable text for {url}")

        paragraphs = cleanup_paragraphs(text, max_paragraphs=max_paragraphs)
        count = word_count(paragraphs)
        if count < MIN_WORDS:
            if tried_fallback:
                raise FetchError(
                    f"{url} does not look like an article "
                    f"(only {count} word(s) after cleanup). "
                    "Try pasting a direct link to the article body."
                )
            raise FetchError(
                f"{url} has only {count} word(s); need at least {MIN_WORDS}"
            )

        return {
            "url": url,
            "title": title,
            "paragraphs": paragraphs,
            "word_count": count,
        }

    raise FetchError(last_error or f"failed to fetch {url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a web article's main text via TinyFish."
    )
    parser.add_argument("url", help="URL of the article to fetch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = fetch_article_text(args.url)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
