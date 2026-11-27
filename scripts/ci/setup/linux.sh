#!/usr/bin/env bash
# Installs what CI needs on a Linux machine (the hosted ubuntu-* runner, or a contributor's machine), only when missing,
# so a job never depends on what happened to be installed by hand. Idempotent; a no-op on a prepared machine.
#
# Packages are installed with apt-get as root: directly when already root, else through passwordless `sudo -n`
# (hosted runners have it). Without it the script fails and names the missing packages instead of hanging on a
# password prompt.
#
# Usage: scripts/ci/setup/linux.sh [--arm64-cross] [--vulkan]
#   (always)       ninja-build, cmake, python3
#   --arm64-cross  g++-aarch64-linux-gnu, qemu-user: the linux-aarch64 cross build and its CTest under qemu (#91)
#   --vulkan       libvulkan-dev, glslc, mesa-vulkan-drivers: the desktop Vulkan add-on module and its test on Mesa's
#                  software lavapipe device (#92)
set -euo pipefail

want=(ninja-build cmake python3)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm64-cross) want+=(g++-aarch64-linux-gnu qemu-user) ;;
    --vulkan) want+=(libvulkan-dev glslc mesa-vulkan-drivers) ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

missing=()
for pkg in "${want[@]}"; do
  dpkg-query -W -f='${Status}' "${pkg}" 2>/dev/null | grep -q 'install ok installed' || missing+=("${pkg}")
done
if [[ ${#missing[@]} -eq 0 ]]; then
  echo "runner setup (linux): everything present (${want[*]})"
  exit 0
fi

if [[ "$(id -u)" -eq 0 ]]; then
  as_root=()
elif sudo -n true 2>/dev/null; then
  as_root=(sudo -n)
else
  echo "runner setup (linux): missing ${missing[*]}, and $(id -un) cannot run sudo without a password." >&2
  echo "Install them once (sudo apt-get install -y ${missing[*]}) or run this script with sudo available." >&2
  exit 1
fi
echo "runner setup (linux): installing ${missing[*]}"
"${as_root[@]}" env DEBIAN_FRONTEND=noninteractive apt-get update -q
"${as_root[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends "${missing[@]}"
