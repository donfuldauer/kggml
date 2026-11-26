#!/usr/bin/env bash
# Builds the samples' Xcode host apps (samples/kggml-showcase/iosApp, samples/whisper-only-minimal/iosApp) with
# xcodebuild, unsigned: for the simulator, for devices (generic iOS, arm64), or both. Needs the native xcframeworks
# (scripts/ios/build-xcframeworks.sh); the Kotlin frameworks are built by the projects' own build phase. Products land
# in build/ios-apps/<sample>/Build/Products. Used by CI (macOS job) and locally (macOS only).
#
# Usage: scripts/ios/build-sample-apps.sh [simulator|device|all] [sample...]
#   sample: kggml-showcase, whisper-only-minimal (default: both)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
kind="${1:-all}"
case "${kind}" in simulator|device|all) ;; *) echo "usage: $0 [simulator|device|all] [sample...]" >&2; exit 2 ;; esac
shift $(($# > 0 ? 1 : 0))
samples=("$@")
((${#samples[@]})) || samples=(kggml-showcase whisper-only-minimal)

destinations=()
[[ "${kind}" != device ]] && destinations+=("generic/platform=iOS Simulator")
[[ "${kind}" != simulator ]] && destinations+=("generic/platform=iOS")

for sample in "${samples[@]}"; do
  project="samples/${sample}/iosApp/iosApp.xcodeproj"
  [[ -d "${project}" ]] || { echo "no Xcode project at ${project}" >&2; exit 2; }
  for destination in "${destinations[@]}"; do
    echo "==== ${sample}: ${destination} ===="
    xcodebuild -project "${project}" -scheme iosApp -configuration Debug -destination "${destination}" \
      -derivedDataPath "build/ios-apps/${sample}" CODE_SIGNING_ALLOWED=NO build -quiet
  done
done
echo "iOS sample apps built (${kind}): ${samples[*]}"
