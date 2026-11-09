#!/usr/bin/env python3
"""Text-only agent loop for debugging the LLM and tools without audio.

Usage:  python scripts/chat_text.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config, tools  # noqa: E402
from src.agent import Agent  # noqa: E402


def main():
    config.require_api_key()
    config.WORKSPACE.mkdir(parents=True, exist_ok=True)
    agent = Agent()
    print(f"Model: {config.NIM_MODEL}. Ctrl+C to quit.")
    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        started = time.time()
        try:
            reply = agent.run_turn(text)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {type(exc).__name__}: {exc}")
            agent.abort("Sorry, I could not reach the language model.")
            continue
        elapsed = int((time.time() - started) * 1000)
        if agent.last_tools:
            print("Tools:", json.dumps(agent.last_tools, ensure_ascii=False))
        print(f"Agent ({elapsed} ms): {reply}")


if __name__ == "__main__":
    main()
