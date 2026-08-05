from pathlib import Path

from scripts.tools.configure_original_e2e_lane_grid import configure_engine


def write_parser(path: Path) -> None:
    path.write_text(
        "class NotTheParser:\n"
        "    pass\n\n"
        "class LaneNNParser(InstanceLaneNNParser):\n"
        "    def __init__(self):\n"
        "        self.CROP_SIZE = 256\n"
        "        self.OVERLAP = 0\n"
        "        self.STEP = 256\n"
        "\n"
        "    def parse(self):\n"
        "        return self.CROP_SIZE, self.STEP\n",
        encoding="utf-8",
    )


def test_configure_engine_updates_only_lane_parser_grid(tmp_path: Path) -> None:
    parser_path = tmp_path / "center_lane_rule" / "parse_nn_output.py"
    parser_path.parent.mkdir(parents=True)
    write_parser(parser_path)

    report = configure_engine(tmp_path, 512)

    changed = parser_path.read_text(encoding="utf-8")
    assert report["changed"] is True
    assert report["patch_size"] == 512
    assert "self.CROP_SIZE = 512" in changed
    assert "self.STEP = 512" in changed
    assert "self.OVERLAP = 0" in changed


def test_configure_engine_is_idempotent(tmp_path: Path) -> None:
    parser_path = tmp_path / "parse_nn_output.py"
    write_parser(parser_path)

    configure_engine(tmp_path, 512)
    report = configure_engine(tmp_path, 512)

    assert report["changed"] is False
