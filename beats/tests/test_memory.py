"""Memory (0.34): the models go when there's nothing to analyse, and ONNX Runtime keeps no arena."""

import os
import sys
import time
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from beats import memory  # noqa: E402
from beats.worker import Worker  # noqa: E402


class Model:
    def __init__(self, loaded=True):
        self.loaded = loaded
        self.unloads = 0

    def unload(self):
        if not self.loaded:
            return False
        self.loaded = False
        self.unloads += 1
        return True


class Events:
    def __init__(self):
        self.said = []

    def event(self, text, kind="info"):
        self.said.append(text)


def worker(last_work, busy=False, **models):
    stub = types.SimpleNamespace(
        last_work=last_work, store=Events(), tracker=models.get("tracker", Model()), style=models.get("style", Model()),
        hearer=models.get("hearer", Model()))
    stub.busy = lambda: busy
    return stub


class LetGoTest(unittest.TestCase):
    def test_models_go_after_ten_minutes_without_a_song(self):
        w = worker(time.time() - memory.MODELS_IDLE - 1)
        self.assertTrue(Worker.let_models_go_if_idle(w))
        self.assertEqual([1, 1, 1], [w.tracker.unloads, w.style.unloads, w.hearer.unloads])
        self.assertIn("Beat This!", w.store.said[0])
        self.assertIn("10 minutes", w.store.said[0])

    def test_they_stay_while_songs_keep_coming(self):
        w = worker(time.time() - 30)
        self.assertFalse(Worker.let_models_go_if_idle(w))
        self.assertEqual(0, w.tracker.unloads)

    def test_they_stay_while_a_song_is_worked_on_however_long(self):
        w = worker(time.time() - 3600, busy=True)
        self.assertFalse(Worker.let_models_go_if_idle(w))
        self.assertEqual(0, w.hearer.unloads)

    def test_nothing_is_said_when_nothing_was_loaded(self):
        w = worker(0, tracker=Model(False), style=Model(False), hearer=None)
        self.assertFalse(Worker.let_models_go_if_idle(w))
        self.assertEqual([], w.store.said)

    def test_a_model_without_unload_is_left_alone(self):
        w = worker(0, tracker=object(), style=None)
        self.assertTrue(Worker.let_models_go_if_idle(w))
        self.assertIn("the models that hear a song", w.store.said[0])
        self.assertNotIn("Beat This!", w.store.said[0])


class HandingBackTest(unittest.TestCase):
    def test_trim_never_fails_and_says_what_is_held(self):
        held = memory.trim()
        self.assertTrue(held is None or held > 0)

    def test_memory_in_words(self):
        self.assertEqual("640 MB", memory.said(640))
        self.assertEqual("7.6 GB", memory.said(7823))
        self.assertEqual("an unknown amount", memory.said(None))

    def test_onnx_runtime_keeps_no_arena(self):
        class Options:
            enable_cpu_mem_arena = True
            enable_mem_pattern = True
            intra_op_num_threads = 0
            inter_op_num_threads = 0

        runtime = types.SimpleNamespace(SessionOptions=Options)
        options = memory.session_options(runtime, threads=3, inter_threads=1)
        self.assertFalse(options.enable_cpu_mem_arena)
        self.assertFalse(options.enable_mem_pattern)
        self.assertEqual((3, 1), (options.intra_op_num_threads, options.inter_op_num_threads))

    def test_the_models_that_hear_let_go_and_keep_their_feelings(self):
        from beats.hear import Hearer

        hearer = Hearer("/nowhere", None, 1)
        self.assertFalse(hearer.unload())
        hearer._sessions, hearer._feelings = ("a", "b", "c"), {"happy": 1}
        self.assertTrue(hearer.unload())
        self.assertFalse(hearer.loaded)
        self.assertEqual({"happy": 1}, hearer._feelings)


if __name__ == "__main__":
    unittest.main()
