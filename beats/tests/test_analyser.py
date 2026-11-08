import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from beats.analyser import resolve_device  # noqa: E402


class DeviceTest(unittest.TestCase):
    def test_auto_takes_the_gpu_when_there_is_one(self):
        self.assertEqual(resolve_device("auto", True, 1), ("cuda", None))
        self.assertEqual(resolve_device("auto", False), ("cpu", None))
        self.assertEqual(resolve_device("", False), ("cpu", None))
        self.assertEqual(resolve_device(None, True, 1), ("cuda", None))

    def test_the_cpu_when_asked(self):
        self.assertEqual(resolve_device("cpu", True, 1), ("cpu", None))
        self.assertEqual(resolve_device(" CPU ", True, 1), ("cpu", None))

    def test_a_gpu_asked_for_but_missing_falls_back_with_a_note(self):
        device, note = resolve_device("cuda", False)
        self.assertEqual(device, "cpu")
        self.assertIn("cuda image", note)

    def test_a_chosen_gpu(self):
        self.assertEqual(resolve_device("cuda:1", True, 2), ("cuda:1", None))
        device, note = resolve_device("cuda:3", True, 2)
        self.assertEqual(device, "cuda")
        self.assertIn("no GPU 3", note)

    def test_an_unknown_device(self):
        device, note = resolve_device("tpu", True, 1)
        self.assertEqual(device, "cpu")
        self.assertIn("isn't known", note)


if __name__ == "__main__":
    unittest.main()
