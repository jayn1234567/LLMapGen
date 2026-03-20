#!/bin/bash
#SBATCH -J stgA_eval1fam
#SBATCH -p gre
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

source /usr/share/lmod/lmod/init/bash
export MODULEPATH=/persist_data/modulefiles:/share/modulefiles
module load conda/3
module load cuda/13.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /file_storage01/home/mingli/project/jn/UniMapGen/.condaenvs/llamafactory-cu128

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
mkdir -p "$ROOT/ckpts/modelscope/Qwen" "$ROOT/outputs" "$ROOT/logs"
mkdir -p "$ROOT/dataset"
JOB_ID="${SLURM_JOB_ID:-manual}"
exec > >(tee -a "$ROOT/logs/stagea_eval_1fam_${JOB_ID}.out") 2> >(tee -a "$ROOT/logs/stagea_eval_1fam_${JOB_ID}.err" >&2)

ln -sfn /file_storage01/home/mingli/data/ckpts/Qwen2___5-VL-3B-Instruct "$ROOT/ckpts/modelscope/Qwen/Qwen2___5-VL-3B-Instruct"
ln -sfn /file_storage01/home/mingli/data/paper16_patch_only_100img_system "$ROOT/dataset/paper16_patch_only_100img_system"
ln -sfn /file_storage01/home/mingli/data/outputs/llamafactory_qwen2_5vl_3b_paper16_patch_only_100img_stdce_8gpu "$ROOT/outputs/llamafactory_qwen2_5vl_3b_paper16_patch_only_100img_lora"
export ADAPTER="$(find /file_storage01/home/mingli/data/outputs/llamafactory_qwen2_5vl_3b_paper16_patch_only_100img_stdce_8gpu -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"

bash "$ROOT/scripts/launch_stagea_patch_only_1fam_eval.sh"
