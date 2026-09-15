#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Deploy on the Linux x86_64 AWS H200 host.' >&2; exit 1; }
for command in docker python3; do
  command -v "$command" >/dev/null || { echo "Missing $command. Use an AWS GPU AMI; --install-runtime installs Docker/toolkit." >&2; exit 1; }
done
docker info >/dev/null
docker compose version >/dev/null
ENDPOINT="${DOCKER_HOST:-$(docker context inspect --format '{{.Endpoints.docker.Host}}')}"
[[ "$ENDPOINT" == unix:///var/run/docker.sock || "$ENDPOINT" == unix:///run/docker.sock ]] || { echo 'Use the local rootful Docker daemon at /var/run/docker.sock; remote/rootless contexts are not supported.' >&2; exit 1; }
if [[ ! -d "$DATA_ROOT" ]]; then
  if [[ "$EUID" == 0 ]]; then
    mkdir -p "$DATA_ROOT"
  else
    sudo install -d -m 0750 -o "$(id -u)" -g "$(id -g)" "$DATA_ROOT"
  fi
fi
if [[ "${1:-}" == --access-only ]]; then exit 0; fi
for command in nvidia-smi flock; do
  command -v "$command" >/dev/null || { echo "Missing $command. Use an AWS GPU AMI." >&2; exit 1; }
done
nvidia-smi --query-gpu=name,memory.total,mig.mode.current --format=csv,noheader,nounits | python3 -c '
import sys
rows = [line.strip().rsplit(",", 2) for line in sys.stdin if line.strip()]
if len(rows) != 8 or any("H200" not in name or int(memory.strip()) < 130000 or mig.strip() == "Enabled" for name, memory, mig in rows):
    raise SystemExit("aws-8xh200 requires 8 H200 GPUs with full memory (no MIG partitioning)")
print("GPU profile verified: 8 × H200")'
python3 scripts/gpu_cleanup.py --report
df -h "$DATA_ROOT"
python3 - <<'PY'
import os, shutil
free = shutil.disk_usage(os.environ['DATA_ROOT']).free / 1024**3
if free < 50:
    raise SystemExit('Less than 50 GiB free in DATA_ROOT. Free space before starting.')
print(f'Data disk free: {free:.0f} GiB. Use a dedicated ~2 TB volume for all weights, images and builds.')
PY
docker run --rm --gpus all nvidia/cuda@sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2 nvidia-smi -L
