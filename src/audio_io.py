"""Microphone capture and speaker playback.

One input stream is opened for the whole session; the callback only stores
frames while ``is_recording`` is true.  This avoids creating/stopping streams
per key press, which is the usual source of deadlocks on PTT release.
"""

import threading

import numpy as np
import sounddevice as sd

from . import config


class Recorder:
    """Persistent mono float32 input stream plus an in-memory buffer."""

    def __init__(self, device=None):
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._frames = 0
        self._recording = False
        self._max_frames = int(config.MAX_RECORDING_S * config.SAMPLE_RATE)
        self._last_status = None
        self._stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=config.AUDIO_INPUT_DEVICE if device is None else device,
            callback=self._callback,
        )
        self._stream.start()

    # -- audio thread -----------------------------------------------------
    def _callback(self, indata, frames, time_info, status):
        if status:
            self._last_status = str(status)
        with self._lock:
            if not self._recording or self._frames >= self._max_frames:
                return
            chunk = indata[:, 0].copy()
            room = self._max_frames - self._frames
            if len(chunk) > room:
                chunk = chunk[:room]
            self._chunks.append(chunk)
            self._frames += len(chunk)

    # -- key-listener / main thread --------------------------------------
    def start(self):
        with self._lock:
            self._chunks = []
            self._frames = 0
            self._recording = True

    def stop(self) -> np.ndarray:
        with self._lock:
            self._recording = False
            chunks = self._chunks
            self._chunks = []
            self._frames = 0
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    def close(self):
        try:
            self._stream.stop()
        finally:
            self._stream.close()

    @property
    def last_status(self):
        return self._last_status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def clip_ok(audio) -> bool:
    """True when a clip is long enough and loud enough to bother transcribing."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size < config.MIN_CLIP_S * config.SAMPLE_RATE:
        return False
    if samples.size == 0:
        return False
    return float(np.sqrt(np.mean(samples**2))) >= config.MIN_RMS


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resampling; enough for speech, and only used when the device
    refuses the requested rate."""
    if src_rate == dst_rate or samples.size == 0:
        return samples
    count = max(1, int(round(len(samples) * dst_rate / src_rate)))
    old_x = np.arange(len(samples), dtype=np.float64)
    new_x = np.linspace(0.0, len(samples) - 1, count)
    return np.interp(new_x, old_x, samples).astype(np.float32)


def _output_device():
    """Explicit config wins; otherwise prefer the PipeWire/Pulse bridge."""
    if config.AUDIO_OUTPUT_DEVICE is not None:
        return config.AUDIO_OUTPUT_DEVICE
    for name in ("pipewire", "pulse"):
        try:
            sd.query_devices(name, kind="output")
            return name
        except Exception:  # noqa: BLE001 - name not present
            continue
    return None


def _default_output_rate() -> int | None:
    try:
        device = _output_device()
        info = (
            sd.query_devices(device, kind="output")
            if device is not None
            else sd.query_devices(kind="output")
        )
        return int(round(float(info["default_samplerate"])))
    except Exception:  # noqa: BLE001 - no output device; let the open fail normally
        return None


def _open_output(requested_rate: int):
    """Open an output stream, falling back to the device's default rate."""
    device = _output_device()
    try:
        stream = sd.OutputStream(
            samplerate=requested_rate,
            channels=1,
            dtype="float32",
            device=device,
        )
        return stream, requested_rate
    except sd.PortAudioError:
        fallback = _default_output_rate()
        if not fallback or fallback == requested_rate:
            raise
        stream = sd.OutputStream(
            samplerate=fallback,
            channels=1,
            dtype="float32",
            device=device,
        )
        return stream, fallback


def play(samples, sample_rate: int, interrupt: threading.Event):
    """Play float32 mono samples in ~50 ms chunks, stopping on ``interrupt``."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return
    samples = np.clip(samples, -1.0, 1.0)
    stream, play_rate = _open_output(sample_rate)
    if play_rate != sample_rate:
        samples = _resample(samples, sample_rate, play_rate)
    chunk = max(1, int(play_rate * 0.05))
    with stream:
        for start in range(0, len(samples), chunk):
            if interrupt.is_set():
                return
            block = samples[start : start + chunk].reshape(-1, 1)
            stream.write(block)
