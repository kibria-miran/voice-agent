import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from src import config, stt, tts

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------


def test_transcribe_rejects_short_clip():
    assert stt.transcribe(np.ones(100, dtype=np.float32)) == ""


def test_transcribe_rejects_silence():
    assert stt.transcribe(np.zeros(config.SAMPLE_RATE, dtype=np.float32)) == ""


def test_transcribe_joins_segments(monkeypatch):
    class Segment:
        def __init__(self, text):
            self.text = text

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            assert kwargs["vad_filter"] is False
            assert kwargs["condition_on_previous_text"] is False
            return iter([Segment(" hello"), Segment(" world ")]), {}

    monkeypatch.setattr(stt, "_model", FakeModel())
    out = stt.transcribe(np.ones(config.SAMPLE_RATE, dtype=np.float32) * 0.1)
    assert out == "hello world"


def test_warm_up_loads_the_model_once(monkeypatch):
    created = []

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def transcribe(self, audio, **kwargs):
            self.calls += 1
            return iter([]), {}

    def factory():
        model = FakeModel()
        created.append(model)
        return model

    monkeypatch.setattr(stt, "_model", None)
    monkeypatch.setattr(stt, "_new_model", factory)
    stt.warm_up()
    stt.warm_up()
    assert len(created) == 1
    assert created[0].calls == 2


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


def test_strip_markdown():
    text = "**Bold** and `code` here.\n\n```gdscript\nprint(1)\n```\n- item"
    out = tts.strip_markdown(text)
    assert "**" not in out
    assert "```" not in out
    assert "print(1)" not in out
    assert "code omitted" in out
    assert "Bold" in out and "item" in out


def test_strip_markdown_keeps_snake_case():
    assert tts.strip_markdown("Call read_file now.") == "Call read_file now."


def test_voice_sample_rate_from_sidecar(tmp_path, monkeypatch):
    voice = tmp_path / "voice.onnx"
    voice.write_bytes(b"x")
    Path(str(voice) + ".json").write_text(json.dumps({"audio": {"sample_rate": 16000}}))
    monkeypatch.setattr(config, "PIPER_VOICE", voice)
    assert tts.voice_sample_rate() == 16000


def test_voice_sample_rate_fallback(tmp_path, monkeypatch):
    voice = tmp_path / "voice.onnx"
    voice.write_bytes(b"x")
    monkeypatch.setattr(config, "PIPER_VOICE", voice)
    assert tts.voice_sample_rate() == config.PIPER_SAMPLE_RATE


def test_split_sentences():
    assert list(tts.split_sentences("One. Two! Three?\nFour")) == [
        "One.",
        "Two!",
        "Three?",
        "Four",
    ]


def _piper_available():
    if not Path(config.PIPER_VOICE).is_file():
        return False
    try:
        import piper  # noqa: F401

        return True
    except ImportError:
        binary = Path(config.PIPER_BIN)
        return binary.is_file() or shutil.which(config.PIPER_BIN) is not None


@pytest.mark.skipif(not _piper_available(), reason="piper or voice model not installed")
def test_tts_produces_audio():
    audio, sample_rate = tts.synthesize("Hello from the test suite.")
    assert audio.size > 0
    assert sample_rate > 0
    assert np.max(np.abs(audio)) > 0


@pytest.mark.skipif(not _piper_available(), reason="piper or voice model not installed")
def test_speak_clears_interrupt(monkeypatch):
    import threading

    from src import audio_io

    played = []
    monkeypatch.setattr(audio_io, "play", lambda audio, sr, interrupt: played.append(len(audio)))
    interrupt = threading.Event()
    interrupt.set()
    tts.speak("Short test.", interrupt)
    assert not interrupt.is_set()
    assert played and played[0] > 0


# ---------------------------------------------------------------------------
# End-to-end speech smoke test (opt-in)
# ---------------------------------------------------------------------------

FIXTURE = ROOT / "tests" / "data" / "hello.wav"
RUN_STT_SMOKE = os.getenv("VOICE_AGENT_STT_SMOKE") == "1"


@pytest.mark.skipif(not FIXTURE.is_file(), reason="no speech fixture")
@pytest.mark.skipif(not RUN_STT_SMOKE, reason="set VOICE_AGENT_STT_SMOKE=1 to run")
def test_stt_transcribes_fixture():
    import wave

    with wave.open(str(FIXTURE), "rb") as wav:
        assert wav.getframerate() == 16000
        raw = wav.readframes(wav.getnframes())
        if wav.getnchannels() == 2:
            raise AssertionError("fixture should be mono")
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    text = stt.transcribe(audio).lower()
    assert "hello" in text
