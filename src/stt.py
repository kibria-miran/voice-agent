"""faster-whisper wrapper (CPU, int8)."""

from . import config  # noqa: F401  MUST stay first: sets thread caps before ctranslate2 loads

import numpy as np

_model = None


def _new_model():
    from faster_whisper import WhisperModel

    return WhisperModel(
        config.WHISPER_MODEL,
        device="cpu",
        compute_type="int8",
        cpu_threads=config.WHISPER_CPU_THREADS,
    )


def _get():
    global _model
    if _model is None:
        _model = _new_model()
    return _model


def warm_up():
    """Load the model and run one real transcription so the first turn is fast."""
    segments, _ = _get().transcribe(
        np.zeros(config.SAMPLE_RATE, dtype=np.float32),
        language="en",
        beam_size=1,
    )
    list(segments)  # the generator is lazy: consume it or nothing runs


def transcribe(audio) -> str:
    if audio is None:
        return ""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size < config.MIN_CLIP_S * config.SAMPLE_RATE:
        return ""
    if float(np.sqrt(np.mean(audio**2))) < config.MIN_RMS:
        return ""
    segments, _ = _get().transcribe(
        audio,
        language="en",
        beam_size=1,
        vad_filter=False,  # PTT already segments the speech
        condition_on_previous_text=False,  # avoids hallucination loops
    )
    return " ".join(seg.text.strip() for seg in segments).strip()
