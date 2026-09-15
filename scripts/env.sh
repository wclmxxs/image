#!/usr/bin/env bash
# Shared host setup. Source only from scripts in this repository.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
if [[ ! -f .env ]]; then
  umask 077
  cp config/env.example .env
fi
IMAGE_LAB_INCOMING_HF_TOKEN="${HF_TOKEN:-}"
set -a
source .env
set +a
if [[ -n "$IMAGE_LAB_INCOMING_HF_TOKEN" ]]; then
  export HF_TOKEN="$IMAGE_LAB_INCOMING_HF_TOKEN"
fi
unset IMAGE_LAB_INCOMING_HF_TOKEN
if [[ -z "${HF_TOKEN:-}" && -f .hf-token.env ]]; then
  source .hf-token.env
fi
export HF_TOKEN="${HF_TOKEN:-}"
export DATA_ROOT="${DATA_ROOT:-/opt/image-lab/data}"
export PORT="${PORT:-18080}"
export BIND_HOST="${BIND_HOST:-127.0.0.1}"
if [[ -z "${API_KEY:-}" ]]; then
  export API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  python3 - <<'PY'
import os, pathlib, re
p = pathlib.Path('.env')
text = p.read_text()
line = 'API_KEY=' + os.environ['API_KEY']
text = re.sub(r'^API_KEY=.*$', line, text, flags=re.M) if re.search(r'^API_KEY=', text, re.M) else text + '\n' + line + '\n'
p.write_text(text)
p.chmod(0o600)
PY
fi
[[ "${#API_KEY}" -ge 24 ]] || { echo 'API_KEY must contain at least 24 characters, or be empty for automatic generation.' >&2; exit 1; }
if [[ "$DATA_ROOT" != /* || "$DATA_ROOT" == "/" || "$DATA_ROOT" == *" "* ]]; then
  echo 'DATA_ROOT must be an absolute path without spaces, and cannot be /.' >&2
  exit 1
fi
export DATA_ROOT="$(python3 -c 'import os,pathlib; print(pathlib.Path(os.environ["DATA_ROOT"]).resolve())')"
if [[ -n "${MAGE_LOCAL_PATH:-}" ]]; then
  [[ "$MAGE_LOCAL_PATH" == /* && -d "$MAGE_LOCAL_PATH" ]] || { echo 'MAGE_LOCAL_PATH must be an existing absolute directory.' >&2; exit 1; }
  export MAGE_LOCAL_PATH="$(python3 -c 'import os,pathlib; print(pathlib.Path(os.environ["MAGE_LOCAL_PATH"]).resolve())')"
fi
