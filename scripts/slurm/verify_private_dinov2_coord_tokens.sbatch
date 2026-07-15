#!/usr/bin/env bash
#SBATCH -J mllm_dino_coord_verify
#SBATCH -p gre
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o slurm_%x_%j.out
#SBATCH -e slurm_%x_%j.err

set -euo pipefail

SCRIPT_PATH=$(readlink -f "$0")
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(readlink -f "$(dirname "${SCRIPT_PATH}")/../..")}}
PYTHON_BIN=${PYTHON_BIN:-/file_storage01/home/mingli/project/jn/UniMapGen/.condaenvs/llamafactory-cu128-dtok-resume-20260326/bin/python3}

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

echo "[verify] host=$(hostname) job_id=${SLURM_JOB_ID:-none}"
echo "[verify] repo=${REPO_ROOT}"
echo "[verify] python=${PYTHON_BIN}"
"${PYTHON_BIN}" -c 'import sys, torch, transformers; print({"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__})'

bash -n scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_private_dinov2_last2_caprl4b_nodeepstack_npu.sh
bash -n scripts/npu/train/train_sft_stage_a_lane_intersection_datasetv2_private_dinov2_last2_caprl4b_coordtokens_nodeepstack_npu.sh

"${PYTHON_BIN}" -m py_compile \
  mllm/coordinate_tokens.py \
  mllm/model/llava_arch.py \
  mllm/model/multimodal_encoder/builder.py \
  mllm/model/multimodal_encoder/dinov2_encoder.py \
  mllm/reward/map_schema.py \
  mllm/train/train_qwen.py \
  scripts/tools/infer_centerline_checkpoint.py \
  scripts/tools/infer_centerline_state_update.py \
  scripts/tools/verify_dinov2_vision_tower.py

"${PYTHON_BIN}" -m unittest -v \
  tests.test_coordinate_tokens \
  tests.test_dinov2_segmentation.Dinov2SegmentationTests.test_mllm_dinov2_wrapper_unfreezes_only_last_two_blocks

"${PYTHON_BIN}" - <<'PY'
from mllm.train.train_qwen import DataArguments, ModelArguments

assert DataArguments.__dataclass_fields__["coordinate_token_mode"].default == "none"
assert DataArguments.__dataclass_fields__["coordinate_token_max"].default == 1000
assert ModelArguments.__dataclass_fields__["mm_vision_unfreeze_last_n_blocks"].default == -1
print("[verify] train_qwen argument contract passed")
PY

echo "[verify] all checks passed"
