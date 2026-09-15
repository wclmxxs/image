#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/scripts/env.sh"
docker compose down
# Also remove workers left behind if the controller was killed before graceful shutdown.
DEPLOYMENT="$(python3 -c 'import hashlib,os,pathlib; print(hashlib.sha256(str(pathlib.Path(os.environ["DATA_ROOT"]).resolve()).encode()).hexdigest()[:12])')"
mapfile -t WORKERS < <(docker ps -aq --filter "label=image-lab.deployment=$DEPLOYMENT")
if [[ "${#WORKERS[@]}" -gt 0 ]]; then docker rm -f "${WORKERS[@]}"; fi
echo 'Stopped this image deployment. Weight cache and test results are preserved.'
