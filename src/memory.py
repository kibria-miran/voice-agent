"""Tiny persistent notes file (optional memory).

Notes live in workspace/notes.md so they fall under the same path safety rules
as every other file the agent touches.
"""

from pathlib import Path

from . import config


def notes_file() -> Path:
    return Path(config.WORKSPACE) / "notes.md"


def load_notes(max_chars: int | None = None) -> str:
    """Return the tail of notes.md, capped to avoid bloating the prompt."""
    path = notes_file()
    if not path.is_file():
        return ""
    limit = max_chars or config.NOTES_MAX_CHARS
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return ""
    text = text.strip()
    if len(text) > limit:
        # Keep the most recent notes.
        text = text[-limit:]
        newline = text.find("\n")
        if newline != -1:
            text = text[newline + 1 :]
        text = "...\n" + text
    return text


def append_note(fact: str) -> str:
    fact = " ".join(fact.split())
    if not fact:
        raise ValueError("empty note")
    path = notes_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"- {fact}\n")
    return fact
