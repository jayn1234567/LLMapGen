#!/usr/bin/env bash
set -euo pipefail

# Download a DI training output root in a layout that this repo can use for
# inference:
#
#   LOCAL_RUN_ROOT/
#     args.json
#     rc_dinov2_centerline_json_modules.pt|pth
#     tokenizer / adapter files
#     checkpoint-29610/
#
# Then run inference with:
#
#   RUN_ROOT=${LOCAL_RUN_ROOT}
#   CHECKPOINT_DIR=${LOCAL_RUN_ROOT}/checkpoint-29610

OBS_CACHE="${OBS_CACHE:-/cache/jn/outputs}"
CHECKPOINT_OBS_PATH="${CHECKPOINT_OBS_PATH:-${checkpoint_OBS_PATH:-obs://yw-ads-model-training-gy1/model-dev/rc-nn/rc_base_model/2026/07/02/642a6e8e97714c558b3681fcd18a03a6/output/20260702_131832}}"
CHECKPOINT_NAMES="${CHECKPOINT_NAMES:-checkpoint-29610}"

OBS_RUN_ROOT="${CHECKPOINT_OBS_PATH%/}"
RUN_ID="${OBS_RUN_ROOT##*/}"
LOCAL_RUN_ROOT="${LOCAL_RUN_ROOT:-${OBS_CACHE}/${RUN_ID}}"

export CHECKPOINT_OBS_PATH
export CHECKPOINT_NAMES
export LOCAL_RUN_ROOT

mkdir -p "${LOCAL_RUN_ROOT}"

echo "============================================================"
echo "[download-di-output] OBS root:        ${CHECKPOINT_OBS_PATH}"
echo "[download-di-output] Local run root:  ${LOCAL_RUN_ROOT}"
echo "[download-di-output] Checkpoints:     ${CHECKPOINT_NAMES}"
echo "============================================================"

python - <<'PY'
import json
import os
from pathlib import Path

try:
    import moxing as mox
except Exception as exc:
    raise SystemExit(f"moxing import failed: {exc!r}. Run this script on ModelArts/Ascend where moxing is available.")


obs_root = os.environ["CHECKPOINT_OBS_PATH"].rstrip("/")
local_root = Path(os.environ["LOCAL_RUN_ROOT"]).expanduser().resolve()
checkpoint_names = [
    item.strip()
    for chunk in os.environ["CHECKPOINT_NAMES"].replace(",", " ").split()
    for item in [chunk]
    if item.strip()
]

root_files = [
    "README.md",
    "readme.md",
    "args.json",
    "adapter_config.json",
    "adapter_model.safetensors",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "mergers.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.config.json",
    "vocab.json",
    "download_checkpoints.sh",
    "rc_dinov2_centerline_json_modules.pt",
    "rc_dinov2_centerline_json_modules.pth",
]

mandatory_groups = {
    "args": ["args.json"],
    "modules": ["rc_dinov2_centerline_json_modules.pt", "rc_dinov2_centerline_json_modules.pth"],
}


def obs_exists(path: str) -> bool:
    try:
        return bool(mox.file.exists(path))
    except Exception:
        return False


def copy_file(src: str, dst: Path) -> bool:
    if not obs_exists(src):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download-di-output] file {src} -> {dst}", flush=True)
    mox.file.copy(src, str(dst))
    return True


def copy_dir(src: str, dst: Path) -> None:
    if not obs_exists(src):
        raise FileNotFoundError(f"OBS checkpoint dir not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[download-di-output] dir  {src} -> {dst}", flush=True)
    mox.file.copy_parallel(src, str(dst))


downloaded_files = []
for name in root_files:
    if copy_file(f"{obs_root}/{name}", local_root / name):
        downloaded_files.append(name)

missing_groups = []
for group_name, candidates in mandatory_groups.items():
    if not any((local_root / name).is_file() for name in candidates):
        missing_groups.append(f"{group_name}: one of {candidates}")
if missing_groups:
    raise FileNotFoundError(
        "Missing required root files after download: " + "; ".join(missing_groups)
    )

downloaded_checkpoints = []
for checkpoint_name in checkpoint_names:
    copy_dir(f"{obs_root}/{checkpoint_name}", local_root / checkpoint_name)
    downloaded_checkpoints.append(checkpoint_name)

manifest = {
    "obs_root": obs_root,
    "local_run_root": str(local_root),
    "downloaded_root_files": downloaded_files,
    "downloaded_checkpoints": downloaded_checkpoints,
}
(local_root / "download_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("============================================================", flush=True)
print("[download-di-output] download complete", flush=True)
print(f"[download-di-output] RUN_ROOT={local_root}", flush=True)
for checkpoint_name in downloaded_checkpoints:
    print(f"[download-di-output] CHECKPOINT_DIR={local_root / checkpoint_name}", flush=True)
print("============================================================", flush=True)
PY

echo "Download complete!"
echo "RUN_ROOT=${LOCAL_RUN_ROOT}"
for checkpoint_name in ${CHECKPOINT_NAMES//,/ }; do
  echo "CHECKPOINT_DIR=${LOCAL_RUN_ROOT}/${checkpoint_name}"
done
