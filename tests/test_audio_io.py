import threading

import numpy as np

from src import audio_io, config


def test_short_clip_rejected():
    assert not audio_io.clip_ok(np.ones(100, dtype=np.float32))


def test_silence_rejected():
    assert not audio_io.clip_ok(np.zeros(config.SAMPLE_RATE, dtype=np.float32))


def test_loud_clip_accepted():
    assert audio_io.clip_ok(np.ones(config.SAMPLE_RATE, dtype=np.float32) * 0.1)


class FakeStream:
    def __init__(self):
        self.writes = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True

    def write(self, block):
        self.writes += 1


def test_play_stops_when_interrupt_is_already_set(monkeypatch):
    stream = FakeStream()
    monkeypatch.setattr(audio_io.sd, "OutputStream", lambda **kw: stream)
    interrupt = threading.Event()
    interrupt.set()
    audio_io.play(np.ones(16000, dtype=np.float32), 16000, interrupt)
    assert stream.writes == 0
    assert stream.closed


def test_play_writes_in_chunks(monkeypatch):
    stream = FakeStream()
    monkeypatch.setattr(audio_io.sd, "OutputStream", lambda **kw: stream)
    interrupt = threading.Event()
    audio_io.play(np.ones(16000, dtype=np.float32), 16000, interrupt)
    assert stream.writes == 20  # 16000 samples at 50 ms per chunk
    assert stream.closed


def test_play_falls_back_to_device_default_rate(monkeypatch):
    streams = []

    def fake_output(**kwargs):
        if kwargs["samplerate"] == 16000:
            raise audio_io.sd.PortAudioError("Invalid sample rate")
        stream = FakeStream()
        streams.append(stream)
        return stream

    monkeypatch.setattr(audio_io.sd, "OutputStream", fake_output)
    monkeypatch.setattr(audio_io, "_default_output_rate", lambda: 48000)
    audio_io.play(np.ones(16000, dtype=np.float32), 16000, threading.Event())
    assert streams and streams[0].writes == 20
    assert streams[0].closed


def test_resample_length():
    out = audio_io._resample(np.ones(16000, dtype=np.float32), 16000, 48000)
    assert len(out) == 48000
