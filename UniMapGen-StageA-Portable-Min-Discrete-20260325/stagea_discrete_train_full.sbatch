#!/bin/bash
#SBATCH -J jn_dtfm
#SBATCH -p gre
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -o logs/stagea_discrete_train_full_%j.out
#SBATCH -e logs/stagea_discrete_train_full_%j.err

set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
bash "$ROOT/scripts/launch_stagea_discrete_train.sh"

