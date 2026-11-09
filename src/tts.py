"""Piper TTS wrapper (CPU, ONNX) plus interruptible playback."""

import json
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from . import audio_io, config

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BOLD_RE = re.compile(r"(\*\*|__)(.*?)\1", re.DOTALL)
_ITALIC_RE = re.compile(r"(?<!\w)([*_])([^*_]+)\1(?!\w)")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)

_voice = None
_voice_error: Exception | None = None


def strip_markdown(text: str) -> str:
    """Make a reply speakable: remove formatting and fenced code."""
    if not text:
        return ""
    text = _FENCE_RE.sub(" (code omitted) ", text)
    text = _IMAGE_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _HEADING_RE.sub("", text)
    text = _BOLD_RE.sub(r"\2", text)
    text = _ITALIC_RE.sub(r"\2", text)
    text = _BULLET_RE.sub("", text)
    text = _NUMBERED_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def voice_sample_rate() -> int:
    sidecar = Path(str(config.PIPER_VOICE) + ".json")
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return int(data["audio"]["sample_rate"])
    except (OSError, ValueError, KeyError, TypeError):
        return config.PIPER_SAMPLE_RATE


def _piper_help() -> str:
    try:
        proc = subprocess.run(
            [config.PIPER_BIN, "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"piper not found ({config.PIPER_BIN!r}). Install with: pip install piper-tts"
        ) from exc


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise RuntimeError(f"unsupported piper WAV sample width: {wav.getsampwidth()}")
        raw = wav.readframes(wav.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def _get_voice():
    """Load the Piper voice once and keep it in memory."""
    global _voice, _voice_error
    if _voice is None:
        if _voice_error is not None:
            raise _voice_error
        voice = Path(config.PIPER_VOICE)
        if not voice.is_file():
            raise FileNotFoundError(
                f"Piper voice missing: {voice}. Run: python scripts/download_voice.py"
            )
        try:
            from piper import PiperVoice

            _voice = PiperVoice.load(str(voice))
        except Exception as exc:  # noqa: BLE001 - fall back to the CLI
            _voice_error = exc
            raise
    return _voice


def warm_up():
    """Load the voice and run one short synthesis (fail-soft)."""
    try:
        voice = _get_voice()
        list(voice.synthesize("Ready."))
    except Exception:  # noqa: BLE001 - speak() reports the real error later
        pass


def _synthesize_api(text: str) -> tuple[np.ndarray, int]:
    voice = _get_voice()
    chunks = list(voice.synthesize(text))
    if not chunks:
        raise RuntimeError("piper produced no audio")
    audio = np.concatenate([chunk.audio_float_array for chunk in chunks])
    return audio.astype(np.float32, copy=False), int(chunks[0].sample_rate)


def _synthesize_cli(text: str) -> tuple[np.ndarray, int]:
    """Fallback: one Piper CLI process per reply."""
    voice = Path(config.PIPER_VOICE)
    if not voice.is_file():
        raise FileNotFoundError(
            f"Piper voice missing: {voice}. Run: python scripts/download_voice.py"
        )

    sample_rate = voice_sample_rate()
    help_text = _piper_help()
    raw_flag = None
    for flag in ("--output-raw", "--output_raw"):
        if flag in help_text:
            raw_flag = flag
            break

    if raw_flag:
        cmd = [config.PIPER_BIN, "--model", str(voice), raw_flag]
        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=120,
            cwd=config.ROOT,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"piper failed ({proc.returncode}): {err}")
        if not proc.stdout:
            raise RuntimeError("piper produced no audio")
        audio = np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0
        return audio, sample_rate

    # Fallback: write a temporary WAV, read it back, delete it.
    output_flag = None
    for flag in ("--output_file", "--output-file", "-f"):
        if flag in help_text:
            output_flag = flag
            break
    output_flag = output_flag or "--output_file"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [config.PIPER_BIN, "--model", str(voice), output_flag, str(tmp_path)]
        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=120,
            cwd=config.ROOT,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"piper failed ({proc.returncode}): {err}")
        with wave.open(str(tmp_path), "rb") as wav:
            sample_rate = wav.getframerate()
        return _read_wav(tmp_path), sample_rate
    finally:
        tmp_path.unlink(missing_ok=True)


def synthesize(text: str) -> tuple[np.ndarray, int]:
    """Return (float32 mono audio, sample_rate), preferring the in-process API."""
    try:
        return _synthesize_api(text)
    except Exception:
        # The CLI fallback is slower (voice reload per call) but works across
        # piper-tts versions.
        return _synthesize_cli(text)


def speak(reply: str, interrupt):
    """Speak a full reply.  This is the only code that clears the interrupt."""
    interrupt.clear()  # I am the speaker now; drop any stale signal
    text = strip_markdown(reply)
    if not text:
        return
    audio, sample_rate = synthesize(text)
    audio_io.play(audio, sample_rate, interrupt)


def split_sentences(text: str):
    """Yield speakable sentences (Phase 3.5 helper)."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    for part in parts:
        part = part.strip()
        if part:
            yield part


def speak_stream(sentences, interrupt):
    """Speak sentence by sentence (Phase 3.5 helper).

    Same ownership rule: clear the event once at entry, never between
    sentences.
    """
    interrupt.clear()
    for sentence in sentences:
        if interrupt.is_set():
            return
        audio, sample_rate = synthesize(sentence)
        audio_io.play(audio, sample_rate, interrupt)
