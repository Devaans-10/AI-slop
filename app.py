"""Gradio front end for the AI slop detector — check any web article.

A text box for a URL and a "Check article" button. On click it fetches the
article's main text with TinyFish, scores each paragraph with the trained
adapter, and renders per-paragraph highlights plus an overall summary.

Requires TINYFISH_API_KEY in the environment and a trained adapter at
checkpoints/run1/adapter.
"""

from __future__ import annotations

import os

import gradio as gr

from slop.fetch_article import fetch_article_text
from slop.predict import analyze, load

ADAPTER_DIR = "checkpoints/run1/adapter"

# Load the model once at startup, not per request.
model, tokenizer = load(ADAPTER_DIR)

COLOR_MAP = {
    "likely AI": "#ff4d4d",
    "unsure": "#ffa64d",
    "likely human": "#4dff4d",
}

REINDEER = (
    "Detectors make mistakes; a score is a signal to look more closely, not "
    "proof that something was written by AI."
)


def label_for_score(score: float) -> str:
    """Map a score to a human-readable label."""
    if score >= 0.7:
        return "likely AI"
    if score >= 0.4:
        return "unsure"
    return "likely human"


def check_article(url: str):
    """Fetch an article, score each paragraph, and return summary + highlights."""
    try:
        result = fetch_article_text(url)
    except Exception as exc:  # noqa: BLE001 - surface fetch errors to the user
        return f"Failed to fetch article: {exc}", []

    paragraphs = result.get("paragraphs") or []
    if not paragraphs:
        return "No extractable article paragraphs found.", []

    text = "\n\n".join(paragraphs)
    analysis = analyze(model, tokenizer, text)
    overall_score = analysis["score"]
    overall_label = label_for_score(overall_score)

    para_scores = [p["score"] for p in analysis["paragraphs"]]
    highlights = [
        (p, label_for_score(s)) for p, s in zip(paragraphs, para_scores)
    ]

    summary = (
        f"Overall score: {overall_score:.3f} ({overall_label})\n"
        f"Word count: {result.get('word_count', 0)}\n"
        f"{REINDEER}"
    )

    return summary, highlights


with gr.Blocks() as demo:
    gr.Markdown("# AI Slop Detector — Check a Web Article")
    gr.Markdown(
        "Paste a URL to check whether the article's paragraphs look "
        "AI-written. Requires TINYFISH_API_KEY and a trained adapter."
    )
    url_input = gr.Textbox(
        label="Article URL",
        placeholder="https://example.com/article",
    )
    check_btn = gr.Button("Check article")
    summary_output = gr.Textbox(label="Result", lines=5)
    highlights_output = gr.HighlightedText(
        label="Paragraph scores", color_map=COLOR_MAP
    )
    check_btn.click(
        fn=check_article,
        inputs=url_input,
        outputs=[summary_output, highlights_output],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
