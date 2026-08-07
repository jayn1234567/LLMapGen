#!/usr/bin/env python3
"""Configure the original E2E evaluator's default IntersectionType filter."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SIGNATURE_PATTERN = re.compile(
    r"(def\s+read_intersections\s*\([^)]*\bonlytype1\s*=\s*)(True|False)(\s*\)\s*:)",
    flags=re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--only-type1", choices=("true", "false"), required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args()


def resolve_utils_path(engine_root: Path) -> Path:
    direct = engine_root / "E2E_EVAL" / "Evaluation" / "utils" / "utils.py"
    candidates = [direct] if direct.is_file() else []
    candidates.extend(
        path
        for path in engine_root.rglob("utils.py")
        if path.parts[-4:] == ("E2E_EVAL", "Evaluation", "utils", "utils.py")
    )
    unique = sorted(set(path.resolve() for path in candidates))
    if len(unique) != 1:
        raise RuntimeError(
            f"Expected exactly one E2E_EVAL/Evaluation/utils/utils.py below {engine_root}, "
            f"found {len(unique)}: {unique}"
        )
    return unique[0]


def configure(engine_root: Path, only_type1: bool, report_json: Path) -> dict:
    engine_root = engine_root.resolve()
    utils_path = resolve_utils_path(engine_root)
    source = utils_path.read_text(encoding="utf-8")
    matches = list(SIGNATURE_PATTERN.finditer(source))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one read_intersections onlytype1 default in {utils_path}, found {len(matches)}"
        )
    if "IntersectionType'] != 1" not in source and 'IntersectionType"] != 1' not in source:
        raise RuntimeError(f"Original IntersectionType filter was not found in {utils_path}")

    before = matches[0].group(2) == "True"
    replacement = "True" if only_type1 else "False"
    updated, count = SIGNATURE_PATTERN.subn(rf"\g<1>{replacement}\g<3>", source, count=1)
    if count != 1:
        raise RuntimeError(f"Unable to configure read_intersections in {utils_path}")
    changed = updated != source
    if changed:
        utils_path.write_text(updated, encoding="utf-8")

    report = {
        "engine_root": str(engine_root),
        "utils_path": str(utils_path),
        "only_type1_before": before,
        "only_type1_after": only_type1,
        "changed": changed,
        "scope": "Both GT and prediction calls that use the original default.",
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    args = parse_args()
    configure(args.engine_root, args.only_type1 == "true", args.report_json)


if __name__ == "__main__":
    main()
