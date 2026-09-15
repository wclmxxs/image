#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/scripts/env.sh"
docker compose ps
exec python3 scripts/client.py models
