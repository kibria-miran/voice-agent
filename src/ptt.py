"""Push-to-talk key handling.

``PTTController`` has no pynput import so it is testable without a keyboard.
``start()`` is the thin pynput wrapper used by main.py.

Interrupt ownership: PTT sets the event, tts.speak() clears it.  main.py never
touches it.
"""

from . import config


class PTTController:
    def __init__(self, recorder, audio_queue, interrupt_event):
        self.recorder = recorder
        self.audio_queue = audio_queue
        self.interrupt = interrupt_event
        self._is_recording = False

    def on_press(self, key=None):
        # Ignore key repeat: only the first press of a hold starts recording.
        if self._is_recording:
            return
        self._is_recording = True
        self.interrupt.set()  # stop any speech in progress
        self.recorder.start()

    def on_release(self, key=None):
        if not self._is_recording:
            return
        self._is_recording = False
        audio = self.recorder.stop()
        if audio is not None and len(audio) > 0:
            self.audio_queue.put(audio)

    @property
    def is_recording(self):
        return self._is_recording


def _resolve_key(name: str):
    """Turn 'f9' / 'space' / 'a' into a pynput key object."""
    from pynput import keyboard

    name = name.strip()
    if not name:
        raise SystemExit("PTT_KEY is empty")
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    try:
        return getattr(keyboard.Key, name)
    except AttributeError:
        raise SystemExit(f"Unknown PTT_KEY {name!r} (try f9, space, ...)")


def start(controller: PTTController):
    """Build and start the global pynput listener, returning it."""
    try:
        from pynput import keyboard
    except Exception as exc:  # pragma: no cover - depends on the desktop session
        raise SystemExit(
            "pynput could not start. PTT needs an X11 session (XFCE on X11), "
            f"not Wayland. Original error: {exc}"
        ) from exc

    target = _resolve_key(config.PTT_KEY)

    def on_press(key):
        if key == target:
            controller.on_press(key)

    def on_release(key):
        if key == target:
            controller.on_release(key)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener
