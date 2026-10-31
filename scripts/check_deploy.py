"""Pre-deployment checks for the AI Slop Detector Hugging Face Space.

Checks that required files exist, requirements.txt includes the needed
packages, README.md has the Hugging Face header, and no project file
contains a leaked API key. Exits with code 1 if any check fails.

Uses only the standard library.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_PACKAGES = ["gradio", "requests", "peft", "transformers", "torch", "google-genai"]

ADAPTER_MODEL_OPTIONS = ["adapter_model.safetensors", "adapter_model.bin"]

# Leaked-key patterns: a provider prefix followed by a key-like value.
# Real API keys are long alphanumeric strings without internal hyphens,
# which avoids matching hyphenated English words (e.g. "risk-management").
SECRET_PATTERNS = [
    re.compile(r"tinyfish-[A-Za-z0-9]{12,}", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9]{20,}", re.IGNORECASE),
]

SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules"}
MAX_FILE_BYTES = 5_000_000


def check_adapter_files() -> bool:
    """Check checkpoints/run1/adapter/ exists with config + model weights."""
    adapter_dir = ROOT / "checkpoints" / "run1" / "adapter"
    all_pass = True

    if not adapter_dir.is_dir():
        print("FAIL: checkpoints/run1/adapter/ directory missing")
        return False

    config = adapter_dir / "adapter_config.json"
    if config.is_file():
        print("PASS: checkpoints/run1/adapter/adapter_config.json exists")
    else:
        print("FAIL: checkpoints/run1/adapter/adapter_config.json missing")
        all_pass = False

    model_found = False
    for model_name in ADAPTER_MODEL_OPTIONS:
        if (adapter_dir / model_name).is_file():
            print(f"PASS: checkpoints/run1/adapter/{model_name} exists")
            model_found = True
            break
    if not model_found:
        print(
            "FAIL: checkpoints/run1/adapter/ missing "
            "adapter_model.safetensors or adapter_model.bin"
        )
        all_pass = False

    return all_pass


def check_basic_files() -> bool:
    """Check that requirements.txt, README.md, and app.py exist."""
    all_pass = True
    for rel in ["requirements.txt", "README.md", "app.py"]:
        path = ROOT / rel
        if path.is_file():
            print(f"PASS: {rel} exists")
        else:
            print(f"FAIL: {rel} missing")
            all_pass = False
    return all_pass


def check_requirements_packages() -> bool:
    """Check requirements.txt includes all required packages."""
    req_path = ROOT / "requirements.txt"
    if not req_path.is_file():
        print("FAIL: requirements.txt missing (cannot verify packages)")
        return False

    content = req_path.read_text(encoding="utf-8")
    all_pass = True
    for pkg in REQUIRED_PACKAGES:
        if pkg.lower() in content.lower():
            print(f"PASS: requirements.txt includes {pkg}")
        else:
            print(f"FAIL: requirements.txt missing {pkg}")
            all_pass = False
    return all_pass


def check_readme_header() -> bool:
    """Check README.md starts with the Hugging Face Spaces YAML header."""
    readme_path = ROOT / "README.md"
    if not readme_path.is_file():
        print("FAIL: README.md missing (cannot verify header)")
        return False

    content = readme_path.read_text(encoding="utf-8")
    all_pass = True

    if not content.startswith("---\n"):
        print("FAIL: README.md does not start with Hugging Face YAML header")
        return False

    end = content.find("\n---\n", 4)
    if end == -1:
        print("FAIL: README.md YAML header block is not closed")
        return False

    header = content[4:end]
    for field in ["title:", "emoji:", "sdk: gradio", "app_file: app.py"]:
        if field in header:
            print(f"PASS: README.md header has {field}")
        else:
            print(f"FAIL: README.md header missing {field}")
            all_pass = False

    return all_pass


def scan_for_secrets() -> bool:
    """Scan all project files for leaked API keys. Returns True if none found."""
    all_pass = True
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for filename in filenames:
            filepath = Path(dirpath) / filename
            try:
                if filepath.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = filepath.read_text(encoding="utf-8", errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue

            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    rel = filepath.relative_to(ROOT)
                    print(f"FAIL: possible leaked key in {rel}")
                    all_pass = False
                    break

    if all_pass:
        print("PASS: no leaked API keys found in project files")
    return all_pass


def main() -> None:
    print("=== Pre-deployment checks ===\n")

    checks = [
        check_adapter_files,
        check_basic_files,
        check_requirements_packages,
        check_readme_header,
        scan_for_secrets,
    ]

    results = []
    for check in checks:
        results.append((check.__name__, check()))

    print("\n=== Summary ===")
    any_fail = False
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {name}")
        if not passed:
            any_fail = True

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
