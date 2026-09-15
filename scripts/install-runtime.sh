#!/usr/bin/env bash
# Explicit opt-in host bootstrap for Ubuntu AWS GPU AMIs; never installs/replaces the driver.
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo 'Requires Linux.' >&2; exit 1; }
source /etc/os-release
[[ "$ID" == ubuntu ]] || { echo 'Automatic runtime installation supports Ubuntu only.' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'Use an AWS GPU AMI with the H200 driver installed.' >&2; exit 1; }
SUDO=()
if [[ "$EUID" -ne 0 ]]; then SUDO=(sudo); fi
if ! command -v docker >/dev/null; then
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y ca-certificates curl
  "${SUDO[@]}" install -m 0755 -d /etc/apt/keyrings
  "${SUDO[@]}" curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  "${SUDO[@]}" chmod a+r /etc/apt/keyrings/docker.asc
  printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu %s stable\n' \
    "$(dpkg --print-architecture)" "$VERSION_CODENAME" | "${SUDO[@]}" tee /etc/apt/sources.list.d/docker.list >/dev/null
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
if ! command -v nvidia-ctk >/dev/null; then
  "${SUDO[@]}" apt-get install -y curl gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    "${SUDO[@]}" gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    "${SUDO[@]}" tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y nvidia-container-toolkit
fi
if [[ -n "$("${SUDO[@]}" docker ps -q)" ]]; then
  echo 'Docker has running containers. Runtime installed; configure NVIDIA runtime during a maintenance window if needed.'
else
  "${SUDO[@]}" nvidia-ctk runtime configure --runtime=docker
  "${SUDO[@]}" systemctl restart docker
fi
if ! docker info >/dev/null 2>&1; then
  echo 'Docker is installed. Run sudo ./start.sh, or grant your login user Docker access and log in again.' >&2
  exit 1
fi
