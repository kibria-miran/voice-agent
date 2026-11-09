#!/usr/bin/env python3
"""Phase 0 environment checks. Prints PASS / WARN / FAIL, never changes anything."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

results = []


def check(status: str, label: str, detail: str = ""):
    results.append(status)
    line = f"[{status}] {label}"
    if detail:
        line += f" - {detail}"
    print(line)


def module(name: str):
    return importlib.util.find_spec(name) is not None


for mod in ["numpy", "sounddevice", "pynput", "openai", "dotenv", "faster_whisper", "piper"]:
    check("PASS" if module(mod) else "FAIL", f"python package {mod}")

if module("sounddevice"):
    import sounddevice as sd

    print("\nAudio devices:")
    try:
        devices = sd.query_devices()
        for index, dev in enumerate(devices):
            if dev["max_input_channels"] or dev["max_output_channels"]:
                print(
                    f"  [{index}] {dev['name']} "
                    f"in={dev['max_input_channels']} out={dev['max_output_channels']} "
                    f"default_sr={dev['default_samplerate']:.0f}"
                )
        native = [
            d
            for d in devices
            if d["max_input_channels"] and int(d["default_samplerate"]) == 16000
        ]
        check(
            "PASS" if native else "WARN",
            "device with native 16000 Hz",
            "" if native else "PortAudio will resample in the callback",
        )
        outputs = [d for d in devices if d["max_output_channels"]]
        check("PASS" if outputs else "FAIL", "output device available")
        print(
            "  The app prefers a 'pipewire'/'pulse' output when present, because "
            "PortAudio's ALSA default is often HDMI."
        )
    except Exception as exc:  # noqa: BLE001
        check("FAIL", "sounddevice query", f"{type(exc).__name__}: {exc}")

print()
piper = None
try:
    from src import config

    piper = config.PIPER_BIN
    check(
        "PASS" if config.NVIDIA_API_KEY else "FAIL",
        "NVIDIA_API_KEY",
        "present" if config.NVIDIA_API_KEY else "missing from .env",
    )
    check(
        "PASS" if Path(config.PIPER_VOICE).is_file() else "FAIL",
        "piper voice",
        str(config.PIPER_VOICE),
    )
    check(
        "PASS" if Path(str(config.PIPER_VOICE) + ".json").is_file() else "FAIL",
        "piper voice config",
        str(config.PIPER_VOICE) + ".json",
    )
except Exception as exc:  # noqa: BLE001
    check("FAIL", "config import", f"{type(exc).__name__}: {exc}")

piper = piper or shutil.which("piper")
if piper:
    try:
        out = subprocess.run([piper, "--help"], capture_output=True, text=True, timeout=15)
        help_text = (out.stdout or "") + (out.stderr or "")
        raw = next((f for f in ("--output-raw", "--output_raw") if f in help_text), None)
        check("PASS", "piper CLI", raw or "no raw-output flag; WAV fallback will be used")
    except Exception as exc:  # noqa: BLE001
        check("FAIL", "piper CLI", f"{type(exc).__name__}: {exc}")
else:
    check("FAIL", "piper CLI", "not in PATH")

godot = shutil.which("godot")
if godot:
    try:
        out = subprocess.run([godot, "--help"], capture_output=True, text=True, timeout=20)
        help_text = (out.stdout or "") + (out.stderr or "")
        check("PASS" if "--check-only" in help_text else "WARN", "godot --check-only")
    except Exception as exc:  # noqa: BLE001
        check("WARN", "godot --check-only", f"{type(exc).__name__}: {exc}")
else:
    check("WARN", "godot", "not in PATH; godot_check will return an error")

zram = Path("/sys/block/zram0/disksize")
if zram.exists():
    try:
        size = int(zram.read_text().strip())
        check("PASS" if size > 0 else "WARN", "zram", f"{size / 1e9:.2f} GB")
    except (OSError, ValueError):
        check("WARN", "zram", "could not read size")
else:
    check("WARN", "zram", "not enabled; install zram-generator")

hard_fail = results.count("FAIL")
print(f"\n{results.count('PASS')} pass, {results.count('WARN')} warn, {hard_fail} fail")
raise SystemExit(1 if hard_fail else 0)
