from pathlib import Path

from scripts.tools.configure_original_e2e_intersection_filter import configure


def test_configure_original_intersection_filter_changes_both_call_default(tmp_path: Path):
    utils_path = tmp_path / "E2E_EVAL/Evaluation/utils/utils.py"
    utils_path.parent.mkdir(parents=True)
    utils_path.write_text(
        "def read_intersections(path, tar_type=None, onlytype1=True):\n"
        "    for feature in []:\n"
        "        if onlytype1 and feature['properties']['IntersectionType'] != 1:\n"
        "            continue\n",
        encoding="utf-8",
    )

    report = configure(tmp_path, False, tmp_path / "report.json")

    assert "onlytype1=False" in utils_path.read_text(encoding="utf-8")
    assert report["only_type1_before"] is True
    assert report["only_type1_after"] is False
    assert report["changed"] is True
