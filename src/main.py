"""Wire everything together and run the push-to-talk loop.

Run from the project root:  python -m src.main
"""

import json
import queue
import sys
import threading
import time
from datetime import datetime, timezone

from . import audio_io, config, llm, ptt, stt, tools, tts
from .agent import Agent


def log_turn(rec: dict):
    config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with config.LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main():
    config.require_api_key()
    config.WORKSPACE.mkdir(parents=True, exist_ok=True)
    config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    agent = Agent()
    audio_q: queue.Queue = queue.Queue()
    interrupt = threading.Event()

    try:
        recorder = audio_io.Recorder()  # main.py owns the recorder lifecycle
    except Exception as exc:  # noqa: BLE001 - give a useful message instead of a traceback
        raise SystemExit(f"Could not open the microphone: {type(exc).__name__}: {exc}")

    controller = ptt.PTTController(recorder, audio_q, interrupt)
    listener = None
    try:
        listener = ptt.start(controller)
        print("Loading speech model...")
        stt.warm_up()
        print("Loading voice...")
        tts.warm_up()
        print("Checking LLM tool calling...")
        llm.self_test_tools(tools.SCHEMAS)

        try:
            tts.speak("Ready.", interrupt)
        except Exception as exc:  # noqa: BLE001 - a silent assistant is still useful
            print(f"WARNING: TTS unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
            log_turn({"ts": _utc_now(), "event": "startup", "tts_error": f"{type(exc).__name__}: {exc}"})

        print(f"Listening. Hold {config.PTT_KEY.upper()} to talk, Ctrl+C to quit.")
        while True:
            try:
                audio = audio_q.get(timeout=0.5)  # timeout keeps Ctrl+C responsive
            except queue.Empty:
                continue

            if not audio_io.clip_ok(audio):
                continue

            t0 = time.time()
            try:
                text = stt.transcribe(audio)
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                print(f"WARNING: STT failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                log_turn(
                    {
                        "ts": _utc_now(),
                        "event": "stt_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if not text:
                continue
            t1 = time.time()

            error = None
            try:
                reply = agent.run_turn(text)
            except Exception as exc:  # noqa: BLE001 - network/API failure
                error = f"{type(exc).__name__}: {exc}"
                print(f"WARNING: LLM failed: {error}", file=sys.stderr)
                reply = "Sorry, I could not reach the language model."
                agent.abort(reply)
            t2 = time.time()

            print(f"You: {text}\nAgent: {reply}")
            tts_error = None
            try:
                tts.speak(reply, interrupt)
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                tts_error = f"{type(exc).__name__}: {exc}"
                print(f"WARNING: TTS failed: {tts_error}", file=sys.stderr)

            log_turn(
                {
                    "ts": _utc_now(),
                    "transcript": text,
                    "stt_ms": int((t1 - t0) * 1000),
                    "llm_ms": int((t2 - t1) * 1000),
                    "tools": agent.last_tools,
                    "tts_ms": int((time.time() - t2) * 1000),
                    "tts_error": tts_error,
                    "error": error,
                    "reply": reply,
                }
            )
    except KeyboardInterrupt:
        pass
    finally:
        if listener is not None:
            listener.stop()
        recorder.close()


if __name__ == "__main__":
    main()
