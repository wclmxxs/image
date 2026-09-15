#!/usr/bin/env bash
set -euo pipefail
PROFILE=aws-8xh200
MODELS=cosmos,flux,ideogram,hunyuan,hunyuan-distil
INSTALL_RUNTIME=0
CHECK_ONLY=0
CHECK_ACCESS_ONLY=0
ASK_HF_TOKEN=0
CLEAN_GPU=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --install-runtime) INSTALL_RUNTIME=1; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    --check-access) CHECK_ACCESS_ONLY=1; shift ;;
    --ask-hf-token) ASK_HF_TOKEN=1; shift ;;
    --no-gpu-cleanup) CLEAN_GPU=0; shift ;;
    --help|-h)
      echo 'Usage: ./start.sh [--profile aws-8xh200] [--models cosmos,flux,ideogram,hunyuan,hunyuan-distil,mage] [--install-runtime] [--check | --check-access] [--ask-hf-token] [--no-gpu-cleanup]'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "$PROFILE" == aws-8xh200 ]] || { echo "Unsupported profile: $PROFILE" >&2; exit 2; }
[[ "$CHECK_ONLY" == 0 || "$CHECK_ACCESS_ONLY" == 0 ]] || { echo 'Choose either --check or --check-access.' >&2; exit 2; }
source "$(dirname "$0")/scripts/env.sh"
if [[ "$ASK_HF_TOKEN" == 1 ]]; then
  [[ -t 0 ]] || { echo '--ask-hf-token requires an interactive terminal.' >&2; exit 2; }
  if ! IFS= read -r -s -p 'Hugging Face read token (hidden): ' HF_TOKEN; then
    printf '\nToken input cancelled.\n' >&2
    exit 2
  fi
  printf '\n' >&2
  [[ -n "$HF_TOKEN" ]] || { echo 'Token must not be empty.' >&2; exit 2; }
  export HF_TOKEN
fi
if [[ "$CHECK_ACCESS_ONLY" == 1 ]]; then
  [[ "$INSTALL_RUNTIME" == 0 ]] || { echo '--check-access cannot install the runtime.' >&2; exit 2; }
  bash scripts/preflight.sh --access-only
else
  if [[ "$INSTALL_RUNTIME" == 1 ]]; then bash scripts/install-runtime.sh; fi
  bash scripts/preflight.sh
  exec 9>"$DATA_ROOT/.start.lock"
  flock -n 9 || { echo 'Another start.sh is running.' >&2; exit 1; }
  mkdir -p "$DATA_ROOT/diagnostics"
fi
export REQUESTED_MODELS="$MODELS"
MODELS="$(python3 - <<'PY'
import json,os
models=json.load(open('config/models.json'))
aliases={alias.lower():key for key,m in models.items() for alias in [key,m['name'],*m['aliases']]}
try:
    print(','.join(dict.fromkeys(aliases[x.strip().lower()] for x in os.environ['REQUESTED_MODELS'].split(','))))
except KeyError as error:
    raise SystemExit(f'Unknown model {error}')
PY
)"
docker build -f docker/controller.Dockerfile -t image-lab/controller:0.1.0 .
PREPARE_ARGS=(--rm --env-file .env -e HF_TOKEN -e "DATA_ROOT=$DATA_ROOT" -e "API_KEY=$API_KEY" -e "MAGE_LOCAL_PATH=${MAGE_LOCAL_PATH:-}" -v "$DATA_ROOT:$DATA_ROOT" -v "$REPO_DIR/config:/app/config:ro")
if [[ -n "${MAGE_LOCAL_PATH:-}" ]]; then
  [[ "$MAGE_LOCAL_PATH" == /* && -d "$MAGE_LOCAL_PATH" ]] || { echo 'MAGE_LOCAL_PATH must be an existing absolute directory.' >&2; exit 1; }
  if [[ "$MAGE_LOCAL_PATH" != "$DATA_ROOT/"* ]]; then PREPARE_ARGS+=(-v "$MAGE_LOCAL_PATH:$MAGE_LOCAL_PATH:ro"); fi
fi
docker run "${PREPARE_ARGS[@]}" image-lab/controller:0.1.0 python -m image_lab.prepare --models "$MODELS" --check-only
if [[ "$CHECK_ACCESS_ONLY" == 1 ]]; then
  echo 'Weight access checks passed. No model weights downloaded; GPU workloads were not inspected or stopped.'
  exit 0
fi
if [[ "$CHECK_ONLY" == 1 ]]; then
  python3 scripts/gpu_cleanup.py --check
  echo 'Host and weight-access checks passed. Models have not been loaded or benchmarked.'
  exit 0
fi
docker run "${PREPARE_ARGS[@]}" image-lab/controller:0.1.0 python -m image_lab.prepare --models "$MODELS"
case ",$MODELS," in
  *,flux,*|*,ideogram,*|*,mage,*) docker build -f docker/base.Dockerfile -t image-lab/base-cu126:0.1.0 . ;;
esac
case ",$MODELS," in
  *,hunyuan,*|*,hunyuan-distil,*) docker build -f docker/base.Dockerfile --build-arg CUDA_IMAGE=nvidia/cuda@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66 -t image-lab/base-cu128:0.1.0 . ;;
esac
BUILT=,
BUILT_BACKENDS=()
IFS=',' read -ra SELECTED <<< "$MODELS"
for model in "${SELECTED[@]}"; do
  backend="$model"
  if [[ "$model" == hunyuan-distil ]]; then backend=hunyuan; fi
  if [[ "$BUILT" == *",$backend,"* ]]; then continue; fi
  BUILD_ARGS=()
  if [[ "$backend" == cosmos ]]; then
    COSMOS_DIGEST=vllm/vllm-omni@sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587
    docker pull "$COSMOS_DIGEST"
    BUILD_ARGS+=(--build-arg "COSMOS_IMAGE=$COSMOS_DIGEST")
  fi
  docker build -f "docker/$backend.Dockerfile" "${BUILD_ARGS[@]}" -t "image-lab/$backend:0.1.0" .
  docker image inspect "image-lab/$backend:0.1.0" --format '{{json .}}' > "$DATA_ROOT/diagnostics/image-$backend.json"
  docker run --rm --entrypoint python3 "image-lab/$backend:0.1.0" -m pip freeze > "$DATA_ROOT/diagnostics/requirements-$backend.txt"
  BUILT_BACKENDS+=("$backend")
  BUILT="$BUILT$backend,"
done
# Finish downloads/builds before taking the GPU node away from its previous workloads.
# Stop the old controller first so it cannot schedule a fresh worker during cleanup.
./stop.sh
if [[ "$CLEAN_GPU" == 1 ]]; then
  CLEANUP_COMMAND=(python3 "$REPO_DIR/scripts/gpu_cleanup.py" --clean --journal "$DATA_ROOT/diagnostics/gpu-cleanup-$(date -u +%Y%m%dT%H%M%SZ).json")
  if [[ "$EUID" == 0 ]]; then
    "${CLEANUP_COMMAND[@]}"
  else
    sudo -n -- "${CLEANUP_COMMAND[@]}" || { echo 'GPU cleanup failed. Read the journal; use sudo ./start.sh if elevated permissions are missing.' >&2; exit 1; }
  fi
else
  python3 scripts/gpu_cleanup.py --check
fi
for backend in "${BUILT_BACKENDS[@]}"; do
  docker run --rm --gpus all --entrypoint python3 "image-lab/$backend:0.1.0" -c \
    'import torch; assert torch.cuda.device_count() == 8; print("CUDA kernel check:", [torch.ones(1, device=f"cuda:{i}").sum().item() for i in range(8)])'
done
# Catch a launcher that restarted a workload while CUDA checks were running.
python3 scripts/gpu_cleanup.py --check
docker compose up -d --force-recreate --wait --wait-timeout 90
echo "API is ready at http://$BIND_HOST:$PORT/docs. API key is stored in $REPO_DIR/.env."
echo 'Models load on first request. Run ./status.sh for availability, then ./lab generate --model flux --prompt "a red fox".'
if [[ ",$MODELS," != *,mage,* ]]; then
  echo 'Mage-Flow-Edit is unavailable until authorized local weights are supplied and start.sh --models mage succeeds.'
fi
