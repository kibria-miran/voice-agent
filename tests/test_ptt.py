import queue
import threading

import numpy as np

from src.ptt import PTTController


class FakeRecorder:
    def __init__(self, audio):
        self.audio = audio
        self.starts = 0
        self.stops = 0

    def start(self):
        self.starts += 1

    def stop(self):
        self.stops += 1
        return self.audio


def make_controller(audio=None):
    recorder = FakeRecorder(audio if audio is not None else np.ones(1600, dtype=np.float32))
    q = queue.Queue()
    interrupt = threading.Event()
    return PTTController(recorder, q, interrupt), recorder, q, interrupt


def test_repeat_presses_are_suppressed():
    controller, recorder, q, interrupt = make_controller()
    controller.on_press()
    controller.on_press()
    controller.on_press()
    assert recorder.starts == 1
    assert interrupt.is_set()
    assert controller.is_recording
    assert q.empty()


def test_release_queues_one_clip():
    controller, recorder, q, interrupt = make_controller()
    controller.on_press()
    controller.on_release()
    assert recorder.stops == 1
    assert not controller.is_recording
    assert q.qsize() == 1
    assert q.get().shape == (1600,)


def test_release_without_press_does_nothing():
    controller, recorder, q, _ = make_controller()
    controller.on_release()
    assert recorder.stops == 0
    assert q.empty()


def test_second_release_does_not_queue_again():
    controller, recorder, q, _ = make_controller()
    controller.on_press()
    controller.on_release()
    controller.on_release()
    assert recorder.stops == 1
    assert q.qsize() == 1
