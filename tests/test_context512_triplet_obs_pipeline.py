import tempfile
import tarfile
import unittest
from pathlib import Path

from scripts.tools.build_context512_roi_triplet_gt_dataset_v2_from_obs_windows import (
    download_tree,
    extract_archives,
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

    def test_tar_archives_are_extracted_and_reused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            payload = root / "payload"
            (payload / "A0_demo").mkdir(parents=True)
            (payload / "A0_demo" / "sample.png").write_bytes(b"png")
            archive = source / "images_000.tar"
            with tarfile.open(archive, "w") as handle:
                handle.add(payload / "A0_demo", arcname="A0_demo")
            extracted = root / "extracted"

            first = extract_archives(source, extracted, workers=2, resume=True)
            second = extract_archives(source, extracted, workers=2, resume=True)

            self.assertEqual(first["archive_count"], 1)
            self.assertEqual(first["extracted_count"], 1)
            self.assertEqual(second["reused_count"], 1)
            target = Path(second["archives"][0]["target"])
            self.assertTrue((target / "A0_demo" / "sample.png").is_file())

    def test_empty_tar_is_recorded_without_stopping_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            archive = source / "empty.tar.gz"
            with tarfile.open(archive, "w:gz"):
                pass
            extracted = root / "extracted"

            first = extract_archives(source, extracted, workers=1, resume=True)
            second = extract_archives(source, extracted, workers=1, resume=True)

            self.assertEqual(first["status"], "passed")
            self.assertEqual(first["empty_count"], 1)
            self.assertEqual(first["archives"][0]["status"], "empty")
            self.assertEqual(second["empty_count"], 1)
            self.assertEqual(second["archives"][0]["status"], "reused_empty")


if __name__ == "__main__":
    unittest.main()
