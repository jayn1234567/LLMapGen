import tempfile
import unittest
from pathlib import Path

from scripts.tools.build_context512_roi_triplet_gt_dataset_v2_from_obs_windows import (
    download_tree,
    normalized_obs_uri,
    read_download_marker,
)


class FakeObsBackend:
    name = "fake"

    def __init__(self):
        self.calls = []

    def download_tree(self, source, destination):
        self.calls.append((source, destination))
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "payload.bin").write_bytes(b"downloaded")


class Context512TripletObsPipelineTest(unittest.TestCase):
    def test_download_marker_makes_resume_idempotent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            destination = root / "download"
            marker = root / ".complete.json"
            source = normalized_obs_uri("obs://bucket/path")
            backend = FakeObsBackend()

            first = download_tree(backend, source, destination, marker, resume=True)
            second = download_tree(backend, source, destination, marker, resume=True)

            self.assertEqual(len(backend.calls), 1)
            self.assertTrue(first["local_files_present"])
            self.assertEqual(first, second)
            self.assertTrue(read_download_marker(marker, source))

    def test_malformed_marker_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            marker = Path(temp_dir) / ".complete.json"
            marker.write_text("{}", encoding="utf-8")
            self.assertFalse(read_download_marker(marker, "obs://bucket/path/"))


if __name__ == "__main__":
    unittest.main()
