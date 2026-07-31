import os
import unittest
from unittest import mock

try:
    from mllm.train import llava_trainer
except ModuleNotFoundError as exc:
    llava_trainer = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class _FakeNpu:
    def __init__(self):
        self.calls = []

    def is_available(self):
        return True

    def synchronize(self):
        self.calls.append("synchronize")

    def empty_cache(self):
        self.calls.append("empty_cache")


@unittest.skipIf(llava_trainer is None, f"training dependencies unavailable: {IMPORT_ERROR}")
class CheckpointMemoryGuardTest(unittest.TestCase):
    def test_cache_release_is_opt_in(self):
        fake_npu = _FakeNpu()
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            llava_trainer.torch, "npu", fake_npu, create=True
        ):
            self.assertFalse(llava_trainer._release_device_cache_before_checkpoint())
        self.assertEqual(fake_npu.calls, [])

    def test_cache_release_synchronizes_before_empty_cache(self):
        fake_npu = _FakeNpu()
        with mock.patch.dict(
            os.environ, {"MLLM_NPU_EMPTY_CACHE_BEFORE_CHECKPOINT": "True"}, clear=True
        ), mock.patch.object(llava_trainer.torch, "npu", fake_npu, create=True), mock.patch.object(
            llava_trainer.gc, "collect"
        ) as collect:
            self.assertTrue(llava_trainer._release_device_cache_before_checkpoint())
        collect.assert_called_once_with()
        self.assertEqual(fake_npu.calls, ["synchronize", "empty_cache"])


if __name__ == "__main__":
    unittest.main()
