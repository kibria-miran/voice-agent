"""Central configuration.

Thread caps MUST be set before ctranslate2 / faster-whisper is imported, so
this module is imported first by every entry point.
"""

import os
import shutil
import sys

# Cap thread pools BEFORE faster-whisper / ctranslate2 is imported.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
# Must pass llm.self_test_tools(); override in .env when it does not.
NIM_MODEL = os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct")

SAMPLE_RATE = 16000
PTT_KEY = os.getenv("PTT_KEY", "f9")  # hold to talk
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny.en")  # try base.en if accuracy is poor
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "2"))
PIPER_VOICE = ROOT / "models" / "en_US-lessac-low.onnx"


def _default_piper() -> str:
    """Prefer piper next to the running interpreter, then PATH."""
    local = Path(sys.executable).with_name("piper")
    if local.is_file():
        return str(local)
    return shutil.which("piper") or "piper"


PIPER_BIN = os.getenv("PIPER_BIN") or _default_piper()


def _device(value: str):
    """Device may be an index or a name; empty means PortAudio's default."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return value


# On PipeWire desktops PortAudio's ALSA default is often HDMI, so the
# play/pause code prefers pipewire/pulse when it exists. Override with an
# index (python -m sounddevice) or a name.
AUDIO_INPUT_DEVICE = _device(os.getenv("AUDIO_INPUT_DEVICE", ""))
AUDIO_OUTPUT_DEVICE = _device(os.getenv("AUDIO_OUTPUT_DEVICE", ""))
PIPER_SAMPLE_RATE = 22050  # fallback; the .onnx.json sidecar is preferred
WORKSPACE = ROOT / "workspace"
LOG_FILE = ROOT / "logs" / "turns.jsonl"

MIN_CLIP_S = 0.3
MAX_RECORDING_S = 60
MIN_RMS = 0.005
MAX_TOKENS = 200
MAX_HISTORY_TURNS = 12
MAX_HISTORY_MESSAGES = 40
MAX_HISTORY_CHARS = 60000
MAX_TOOL_ROUNDS = 5
MAX_TOOL_RESULT_CHARS = 8000
MAX_TOOL_CHARS_PER_TURN = 16000
LIST_MAX_ENTRIES = 200
READ_FILE_BYTES = 8192
MAX_WRITE_CHARS = 200_000
COMMAND_TIMEOUT_S = 20
NOTES_MAX_CHARS = 2000
LLM_RETRIES = 2
LLM_RETRY_BASE_S = 0.75
DEBUG = os.getenv("DEBUG") == "1"

SYSTEM_PROMPT = (
    "Concise voice assistant for a Godot 4 dev. Reply in 1-2 plain sentences, no markdown. "
    "Write code to files with write_file; don't read code aloud."
)


def build_system_prompt(notes: str = "") -> str:
    """Return the system prompt, optionally with remembered notes appended."""
    prompt = SYSTEM_PROMPT
    if notes.strip():
        prompt += "\n\nThings the user asked you to remember:\n" + notes.strip()
    return prompt


def require_api_key():
    """Called only by main.py so tests that never touch the LLM run without a key."""
    if not NVIDIA_API_KEY:
        raise SystemExit("NVIDIA_API_KEY missing from .env")
