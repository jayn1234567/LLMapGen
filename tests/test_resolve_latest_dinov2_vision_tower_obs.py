import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "tools"
    / "resolve_latest_dinov2_vision_tower_obs.py"
)
SPEC = importlib.util.spec_from_file_location("resolve_latest_dinov2_obs", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeMoxFile:
    def __init__(self, root: Path):
        self.root = root

    def _local(self, obs_path: str) -> Path:
        assert obs_path.startswith("obs://bucket/")
        return self.root / obs_path.removeprefix("obs://bucket/")

    def exists(self, path: str) -> bool:
        return self._local(path).exists()

    def list_directory(self, path: str):
        local = self._local(path)
        return [child.name for child in local.iterdir()]

    def copy(self, source: str, target: str):
        shutil.copyfile(self._local(source), target)


class FakeMox:
    def __init__(self, root: Path):
        self.file = FakeMoxFile(root)


class ResolveLatestDinov2VisionTowerTests(unittest.TestCase):
    def _make_run(self, registry: Path, name: str, completed: float, *, complete: bool = True):
        run = registry / name
        for relative in MODULE.REQUIRED_RUN_ARTIFACTS:
            if relative == MODULE.SUCCESS_MARKER:
                continue
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        marker = {
            "status": "passed",
            "completed_unix_time": completed,
            "best_metric": "lane_iou",
            "best_metric_value": 0.75,
        }
        (run / MODULE.SUCCESS_MARKER).write_text(json.dumps(marker), encoding="utf-8")
        if not complete:
            (run / "best" / "vision_tower" / "model.safetensors").unlink()
        return run

    def test_selects_latest_completed_run_and_its_internal_best(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = Path(temp_dir) / "registry"
            registry.mkdir()
            self._make_run(registry, "run_old", 100.0)
            self._make_run(registry, "run_latest", 300.0)
            self._make_run(registry, "run_incomplete", 400.0, complete=False)

            report = MODULE.resolve_latest(FakeMox(Path(temp_dir)), "obs://bucket/registry")

            selected = report["selected"]
            self.assertEqual(selected["run_root"], "obs://bucket/registry/run_latest")
            self.assertEqual(
                selected["vision_tower"],
                "obs://bucket/registry/run_latest/best/vision_tower",
            )
            rejected = {item["run_root"]: item for item in report["candidates"]}
            self.assertFalse(rejected["obs://bucket/registry/run_incomplete"]["complete"])

    def test_rejects_registry_without_complete_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = Path(temp_dir) / "registry"
            registry.mkdir()
            self._make_run(registry, "broken", 100.0, complete=False)
            with self.assertRaisesRegex(RuntimeError, "No completed DINOv2"):
                MODULE.resolve_latest(FakeMox(Path(temp_dir)), "obs://bucket/registry")


if __name__ == "__main__":
    unittest.main()
