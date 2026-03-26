#!/bin/bash
#SBATCH -J jn_dtev
#SBATCH -p gre
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/stagea_discrete_eval_%j.out
#SBATCH -e logs/stagea_discrete_eval_%j.err

set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
bash "$ROOT/scripts/launch_stagea_discrete_eval.sh"
