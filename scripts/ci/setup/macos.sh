#!/usr/bin/env bash
# Installs what CI needs on a macOS runner (hosted macos-* or the owner's self-hosted local-mac), only when missing.
# Idempotent; a no-op on a prepared machine.
#
# Command-line tools come from Homebrew, which must NOT run as root (Homebrew refuses), so this runs as the runner
# user. Xcode itself cannot be installed unattended (App Store / Apple ID), so it is only checked: a missing Xcode
# or iOS simulator runtime fails here with the fix instead of deep inside a build.
#
# Usage: scripts/ci/setup/macos.sh
set -euo pipefail

[[ "$(id -u)" -ne 0 ]] || { echo "runner setup (macos): do not run as root (Homebrew refuses root)" >&2; exit 1; }

missing=()
for tool in ninja cmake; do command -v "${tool}" >/dev/null 2>&1 || missing+=("${tool}"); done
if [[ ${#missing[@]} -gt 0 ]]; then
  command -v brew >/dev/null 2>&1 || {
    echo "runner setup (macos): missing ${missing[*]} and Homebrew is not installed (https://brew.sh)" >&2; exit 1; }
  echo "runner setup (macos): installing ${missing[*]} with Homebrew"
  HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1 brew install "${missing[@]}"
fi

xcodebuild -version >/dev/null 2>&1 || {
  echo "runner setup (macos): Xcode is not installed or not selected (install it, then: sudo xcode-select -s /Applications/Xcode.app)" >&2
  exit 1
}
xcrun simctl list runtimes 2>/dev/null | grep -q '^iOS ' || {
  echo "runner setup (macos): no iOS simulator runtime (install one: xcodebuild -downloadPlatform iOS)" >&2
  exit 1
}
echo "runner setup (macos): ninja $(ninja --version), $(cmake --version | head -n1), $(xcodebuild -version | head -n1)"
