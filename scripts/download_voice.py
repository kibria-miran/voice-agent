#!/usr/bin/env python3
"""Download a Piper voice (.onnx + .onnx.json) from rhasspy/piper-voices.

Usage:
    python scripts/download_voice.py [--voice en_US-lessac] [--quality low]
"""

import argparse
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VOICE = "en_US-lessac"
DEFAULT_QUALITY = "low"


def voice_url(voice: str, quality: str, suffix: str) -> str:
    # en_US-lessac -> en / en_US / lessac
    try:
        lang, rest = voice.split("_", 1)
        region, name = rest.split("-", 1)
    except ValueError as exc:
        raise ValueError(f"voice must look like en_US-lessac, got {voice!r}") from exc
    locale = f"{lang}_{region}"
    return f"{BASE}/{lang}/{locale}/{name}/{quality}/{voice}-{quality}.onnx{suffix}"


def download(url: str, dest: Path, force: bool = False):
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"skip (exists): {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "voice-agent/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        tmp = dest.with_suffix(dest.suffix + ".part")
        done = 0
        with tmp.open("wb") as fh:
            while True:
                block = response.read(1024 * 256)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if total:
                    pct = done * 100 // total
                    print(f"\r  {done / 1e6:6.1f} / {total / 1e6:6.1f} MB ({pct}%)", end="")
                else:
                    print(f"\r  {done / 1e6:6.1f} MB", end="")
        print()
    tmp.replace(dest)
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--quality", default=DEFAULT_QUALITY, choices=["low", "medium", "high"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    models = ROOT / "models"
    onnx = models / f"{args.voice}-{args.quality}.onnx"
    try:
        download(voice_url(args.voice, args.quality, ""), onnx, args.force)
        download(voice_url(args.voice, args.quality, ".json"), Path(str(onnx) + ".json"), args.force)
    except Exception as exc:  # noqa: BLE001 - CLI error message
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"\nDone. Voice path: {onnx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
