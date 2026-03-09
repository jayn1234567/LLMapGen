#!/usr/bin/env bash
set -euo pipefail

source /home/lenovo/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/data/project/jn/UniMapGen/.envs/unimapgen-gpu

sanitize_ld_library_path() {
  local old_path="${1:-}"
  local new_parts=()
  local part=""
  local seen=""
  IFS=':' read -r -a parts <<< "${old_path}"
  for part in "${parts[@]}"; do
    [[ -z "${part}" ]] && continue
    case "${part}" in
      /usr/local/cuda*|*/cuda/lib64|*/cuda-*/lib64)
        continue
        ;;
    esac
    case ":${seen}:" in
      *":${part}:"*) continue ;;
      *) seen="${seen}:${part}"; new_parts+=("${part}") ;;
    esac
  done
  if [[ ${#new_parts[@]} -gt 0 ]]; then
    printf '%s' "${new_parts[0]}"
    for part in "${new_parts[@]:1}"; do
      printf ':%s' "${part}"
    done
  fi
}

SANITIZED_LD_LIBRARY_PATH="$(sanitize_ld_library_path "${LD_LIBRARY_PATH:-}")"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
while IFS= read -r sitepkg; do
  [[ -z "${sitepkg}" ]] && continue
  for libdir in "${sitepkg}"/cusparselt/lib \
                "${sitepkg}"/nvidia/*/lib \
                "${sitepkg}"/nvidia/cu*/lib; do
    if [[ -d "${libdir}" ]]; then
      export LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH}"
    fi
  done
done < <(python - <<'PY'
import site
for path in site.getsitepackages():
    if "site-packages" in path:
        print(path)
PY
)
if [[ -n "${SANITIZED_LD_LIBRARY_PATH}" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${SANITIZED_LD_LIBRARY_PATH}"
fi
unset CUDA_HOME
unset CUDA_PATH

echo "Activated env: /mnt/data/project/jn/UniMapGen/.envs/unimapgen-gpu"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
python -c "import sys; print('python=', sys.executable)"

if [[ "${UNIMAPGEN_SKIP_TORCH_PROBE:-0}" == "1" ]]; then
  echo "[Info] Skipping torch/CUDA probe because UNIMAPGEN_SKIP_TORCH_PROBE=1"
  return 0 2>/dev/null || exit 0
fi

if command -v timeout >/dev/null 2>&1; then
  if ! timeout 20s python -c "import torch; print('torch=', torch.__version__); print('cuda_built=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"; then
    echo "[Warn] Torch/CUDA probe timed out after 20s; continuing without blocking startup."
  fi
else
  python -c "import torch; print('torch=', torch.__version__); print('cuda_built=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
fi
