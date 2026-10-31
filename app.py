"""AI Slop Detector backend — FastAPI + static login checker.

Replaces the Gradio front end with a single static HTML page served from
``static/``. All detection logic in ``slop/`` is unchanged.

Endpoints:
    POST /login        session-based login (JSON username/password)
    POST /logout       clear the session
    GET  /me           { "logged_in": bool }
    POST /analyze      same request/response as before, requires login
    GET  /             static index.html (html=True)
"""

from __future__ import annotations

import asyncio
import hmac
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from slop.fetch_article import fetch_article_text
from slop.predict import analyze, load

ADAPTER_DIR = "checkpoints/run1/adapter"

# --- Required environment variables (fail fast at startup) ------------------
APP_USERNAME = os.environ.get("APP_USERNAME")
APP_PASSWORD = os.environ.get("APP_PASSWORD")
SESSION_SECRET = os.environ.get("SESSION_SECRET")

_missing = [
    name
    for name, value in (
        ("APP_USERNAME", APP_USERNAME),
        ("APP_PASSWORD", APP_PASSWORD),
        ("SESSION_SECRET", SESSION_SECRET),
    )
    if not value
]
if _missing:
    raise RuntimeError(
        "Missing required environment variable(s): " + ", ".join(_missing)
    )

# --- Model (loaded once at startup) -----------------------------------------
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


# --- FastAPI app ------------------------------------------------------------
api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_URL", "*")],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

api.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",
    https_only=False,
)


@api.post("/login")
async def login(request: Request):
    """JSON { username, password }. Constant-time compare, 1s delay on failure."""
    body = await request.json()
    username = body.get("username", "")
    password = body.get("password", "")

    ok = (
        hmac.compare_digest(username, APP_USERNAME)
        and hmac.compare_digest(password, APP_PASSWORD)
    )
    if ok:
        request.session["logged_in"] = True
        return JSONResponse({"ok": True})

    await asyncio.sleep(1)
    return JSONResponse(
        {"ok": False, "error": "Wrong username or password"},
        status_code=401,
    )


@api.post("/logout")
def logout(request: Request):
    request.session.clear()
    return JSONResponse({"ok": True})


@api.get("/me")
def me(request: Request):
    return JSONResponse({"logged_in": bool(request.session.get("logged_in"))})


@api.post("/analyze")
async def analyze_endpoint(request: Request):
    if not request.session.get("logged_in"):
        return JSONResponse(
            {"ok": False, "error": "Not logged in"}, status_code=401
        )
    body = await request.json()
    url = body.get("url", "")
    summary, highlights = check_article(url)
    return JSONResponse({"summary": summary, "highlights": highlights})


# Main app: API routes first, static catch-all mount last.
app = FastAPI()
app.mount("/api", api)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
