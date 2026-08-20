"""Exact task-text normalisation; fuzzy matching is intentionally forbidden."""

from __future__ import annotations

import re
import unicodedata


_WHITESPACE = re.compile(r"\s+")


def normalize_task_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving exact wording/case."""

    normalized = unicodedata.normalize("NFC", text)
    return _WHITESPACE.sub(" ", normalized.strip())


def task_text_matches(left: str, right: str) -> bool:
    return normalize_task_text(left) == normalize_task_text(right)
