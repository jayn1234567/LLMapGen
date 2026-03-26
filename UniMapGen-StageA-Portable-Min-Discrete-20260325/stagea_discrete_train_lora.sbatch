#!/bin/bash
#SBATCH -J jn_dtlr
#SBATCH -p gre
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -o logs/stagea_discrete_train_lora_%j.out
#SBATCH -e logs/stagea_discrete_train_lora_%j.err

set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
bash "$ROOT/scripts/launch_stagea_discrete_train_lora.sh"

