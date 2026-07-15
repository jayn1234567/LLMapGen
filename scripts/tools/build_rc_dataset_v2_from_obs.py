#!/usr/bin/env python3
"""Download the seven RC raw sources and build paired Dataset V2 assets.

MoXing is imported only when a download or upload stage is requested, so the
same entrypoint can be smoke-tested locally with --skip-download --skip-upload.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


DEFAULT_SOURCE_OBS_ROOTS = [
    "obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_0902_1935/",
    "obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_0426_1639/",
    "obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_0922_0901/",
    "obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_1013_2100/",
    "obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_1023_2143/",
    "obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_1029_1153/",
    "obs://yw-ncasd-result-gy1/data/RCDataset/BaseModel/rc_airflow_task_1120_2889/",
]
DEFAULT_OUTPUT_OBS_ROOT = (
    "obs://yw-ads-training-gy1/data/external/personal/h58801830/whu/jn/data/"
    "rc_dataset_v2_550k_noempty_i30_shift128/"
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-obs-root",
        action="append",
        default=[],
        help="Repeat to override the built-in seven OBS roots.",
    )
    parser.add_argument(
        "--work-root",
        default="",
        help="Required on Windows; choose a short path on a large NTFS disk, for example D:\\rcv2.",
    )
    parser.add_argument("--raw-root", default="", help="Default: WORK_ROOT/raw_sources")
    parser.add_argument("--output-root", default="", help="Default: WORK_ROOT/output")
    parser.add_argument(
        "--output-obs-root",
        default=DEFAULT_OUTPUT_OBS_ROOT,
        help=f"Destination OBS directory. Default: {DEFAULT_OUTPUT_OBS_ROOT}",
    )
    parser.add_argument("--views", choices=["both", "local", "context"], default="both")
    parser.add_argument("--train-target-samples", type=int, default=550000)
    parser.add_argument(
        "--train-stride",
        type=int,
        default=128,
        help="Train crop stride; 128 enables half-patch translated windows as a unique-data fallback.",
    )
    parser.add_argument("--difficulty-ratios", default="empty=0,easy=0.30,medium=0.33,hard=0.27,very_hard=0.10")
    parser.add_argument("--intersection-target-ratio", type=float, default=0.30)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--difficulty-seed", type=int, default=20260713)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--duplicate-policy", choices=["last", "first", "error"], default="last")
    parser.add_argument("--upload-mode", choices=["tar", "directory"], default="tar")
    parser.add_argument("--obs-backend", choices=["auto", "moxing", "obsutil"], default="auto")
    parser.add_argument("--obsutil-path", default="", help="Path to obsutil.exe. Default: find obsutil on PATH.")
    parser.add_argument("--obsutil-config", default="", help="Optional obsutil config file path.")
    parser.add_argument("--obsutil-jobs", type=int, default=8, help="Parallel jobs for recursive obsutil copies.")
    parser.add_argument(
        "--remove-package-after-upload",
        action="store_true",
        help="Delete each generated tar after a successful upload to reduce peak disk usage.",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed downloads, packages, and existing PNG files.")
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument(
        "--archive-workers",
        type=int,
        default=16,
        help="Number of independent .tar.gz archives to extract concurrently.",
    )
    return parser.parse_args(argv)


def import_moxing():
    try:
        import moxing as mox
    except Exception as exc:
        raise RuntimeError(
            "MoXing is required for OBS download/upload. Run in the DI/Ascend environment "
            "where moxing-framework is installed, or use --skip-download --skip-upload."
        ) from exc
    if not hasattr(mox, "file"):
        raise RuntimeError(
            "The imported 'moxing' package has no mox.file API. Uninstall the unrelated PyPI package named "
            "moxing and install Huawei moxing-framework, or use --obs-backend obsutil."
        )
    return mox


class MoxingBackend:
    name = "moxing"

    def __init__(self):
        self.mox = import_moxing()

    def download_tree(self, source, destination):
        self.mox.file.copy_parallel(source, str(destination))

    def upload_tree(self, source, destination):
        self.mox.file.copy_parallel(str(source), destination)

    def upload_file(self, source, destination):
        self.mox.file.copy(str(source), destination)


class ObsutilBackend:
    name = "obsutil"

    def __init__(self, executable, config_path="", jobs=8):
        self.executable = str(executable)
        self.config_path = str(config_path or "")
        self.jobs = max(1, int(jobs))

    def _run(self, arguments, recursive=False):
        command = [self.executable, *arguments]
        if recursive:
            command.extend(["-r", "-f", f"-j={self.jobs}"])
        else:
            command.append("-f")
        if self.config_path:
            command.append(f"-config={self.config_path}")
        print("[dataset-v2] obsutil command:", shlex.join(command), flush=True)
        subprocess.run(command, check=True)

    def download_tree(self, source, destination):
        self._run(["cp", source, str(destination)], recursive=True)

    def upload_tree(self, source, destination):
        self._run(["cp", str(source), destination], recursive=True)

    def upload_file(self, source, destination):
        self._run(["cp", str(source), destination], recursive=False)


def resolve_obs_backend(args):
    requested = args.obs_backend
    obsutil = args.obsutil_path or shutil.which("obsutil") or shutil.which("obsutil.exe")
    if requested in {"auto", "obsutil"} and obsutil:
        return ObsutilBackend(obsutil, args.obsutil_config, args.obsutil_jobs)
    if requested == "obsutil":
        raise FileNotFoundError(
            "obsutil was requested but not found. Add obsutil.exe to PATH or pass --obsutil-path."
        )
    if requested in {"auto", "moxing"}:
        return MoxingBackend()
    raise ValueError(f"unsupported OBS backend: {requested}")


def source_name(uri):
    return uri.rstrip("/").rsplit("/", 1)[-1]


def download_sources(sources, raw_root, resume, backend):
    local_roots = []
    for index, source in enumerate(sources):
        local_root = raw_root / f"{index:02d}_{source_name(source)}"
        marker = local_root / ".obs_download_complete.json"
        if resume and marker.exists():
            print(f"[dataset-v2] reuse completed download: {local_root}", flush=True)
            local_roots.append(local_root)
            continue
        local_root.mkdir(parents=True, exist_ok=True)
        print(f"[dataset-v2] download {source} -> {local_root}", flush=True)
        backend.download_tree(source, local_root)
        marker.write_text(
            json.dumps(
                {"source": source, "local_root": str(local_root), "obs_backend": backend.name},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        local_roots.append(local_root)
    return local_roots


def expected_local_roots(sources, raw_root):
    return [raw_root / f"{index:02d}_{source_name(source)}" for index, source in enumerate(sources)]


def run_builder(args, sources, local_roots, output_root):
    builder = REPO_ROOT / "data_process" / "build_dataset_v2.py"
    command = [
        sys.executable,
        str(builder),
        "--output-root",
        str(output_root),
        "--views",
        args.views,
        "--train-target-samples",
        str(args.train_target_samples),
        "--train-stride",
        str(args.train_stride),
        "--difficulty-ratios",
        args.difficulty_ratios,
        "--intersection-target-ratio",
        str(args.intersection_target_ratio),
        "--split-seed",
        str(args.split_seed),
        "--difficulty-seed",
        str(args.difficulty_seed),
        "--duplicate-policy",
        args.duplicate_policy,
        "--archive-workers",
        str(args.archive_workers),
    ]
    for source, local_root in zip(sources, local_roots):
        command.extend(["--input-root", str(local_root), "--source-uri", source])
    if args.limit_samples is not None:
        command.extend(["--limit-samples", str(args.limit_samples)])
    if args.keep_archives:
        command.append("--keep-archives")
    if args.resume:
        command.append("--skip-existing-images")
    print("[dataset-v2] build command:", shlex.join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def create_variant_tar(variant_root, package_path, resume):
    if resume and package_path.exists() and package_path.stat().st_size > 0:
        print(f"[dataset-v2] reuse package: {package_path}", flush=True)
        return
    package_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = package_path.with_suffix(package_path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    print(f"[dataset-v2] package {variant_root} -> {package_path}", flush=True)
    with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as archive:
        archive.add(variant_root, arcname=variant_root.name, recursive=True)
    temporary.replace(package_path)


def upload_outputs(
    output_root,
    output_obs_root,
    views,
    upload_mode,
    resume,
    backend,
    remove_package_after_upload,
):
    variants = []
    if views in {"both", "local"}:
        variants.append("local256")
    if views in {"both", "context"}:
        variants.append("context512_roi256")
    obs_root = output_obs_root.rstrip("/")
    if upload_mode == "directory":
        for variant in variants:
            source = output_root / variant
            destination = f"{obs_root}/{variant}"
            print(f"[dataset-v2] upload directory {source} -> {destination}", flush=True)
            backend.upload_tree(source, destination)
    else:
        package_root = output_root / "packages"
        for variant in variants:
            package = package_root / f"{variant}.tar"
            create_variant_tar(output_root / variant, package, resume)
            destination = f"{obs_root}/{package.name}"
            print(f"[dataset-v2] upload file {package} -> {destination}", flush=True)
            backend.upload_file(package, destination)
            if remove_package_after_upload:
                package.unlink()
                print(f"[dataset-v2] removed uploaded package: {package}", flush=True)
    for filename in ("build_summary.json", "split_manifest.json", "manifests/balance_report.json"):
        source = output_root / filename
        if source.exists():
            backend.upload_file(source, f"{obs_root}/{Path(filename).name}")


def main(argv=None):
    args = parse_args(argv)
    sources = args.source_obs_root or list(DEFAULT_SOURCE_OBS_ROOTS)
    if not args.work_root:
        if os.name == "nt":
            raise ValueError(
                "--work-root is required on Windows. Use a short path on a large NTFS disk, for example D:\\rcv2."
            )
        args.work_root = "/cache/rc_dataset_v2"
    work_root = Path(args.work_root)
    raw_root = Path(args.raw_root) if args.raw_root else work_root / "raw_sources"
    output_root = Path(args.output_root) if args.output_root else work_root / "output"
    raw_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[dataset-v2] source count: {len(sources)}", flush=True)
    for index, source in enumerate(sources):
        print(f"[dataset-v2] source[{index}]: {source}", flush=True)
    print(f"[dataset-v2] raw root: {raw_root}", flush=True)
    print(f"[dataset-v2] output root: {output_root}", flush=True)
    disk = shutil.disk_usage(work_root)
    print(f"[dataset-v2] work disk free: {disk.free / (1024 ** 3):.1f} GiB", flush=True)

    needs_backend = (not args.skip_download) or (
        not args.skip_upload and bool(args.output_obs_root)
    )
    backend = resolve_obs_backend(args) if needs_backend else None
    if backend is not None:
        print(f"[dataset-v2] OBS backend: {backend.name}", flush=True)

    if args.skip_download:
        local_roots = expected_local_roots(sources, raw_root)
        missing = [str(path) for path in local_roots if not path.exists()]
        if missing:
            raise FileNotFoundError(f"--skip-download was set but local source roots are missing: {missing}")
    else:
        local_roots = download_sources(sources, raw_root, args.resume, backend)

    if not args.skip_build:
        run_builder(args, sources, local_roots, output_root)

    if not args.skip_upload:
        if not args.output_obs_root:
            print("[dataset-v2] output OBS root is empty; skip upload", flush=True)
        else:
            upload_outputs(
                output_root,
                args.output_obs_root,
                args.views,
                args.upload_mode,
                args.resume,
                backend,
                args.remove_package_after_upload,
            )
    print("[dataset-v2] done", flush=True)


if __name__ == "__main__":
    main()
