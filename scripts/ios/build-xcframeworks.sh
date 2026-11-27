#!/usr/bin/env bash
# Builds the native iOS xcframeworks the samples' Xcode host apps embed, into build/native-ios (macOS only):
#   all (default)   GgmlRuntime, KggmlWhisper, KggmlLlama, KggmlDiffusion   (samples/kggml-showcase/iosApp)
#   whisper         GgmlRuntime, KggmlWhisper                                (samples/whisper-only-minimal/iosApp)
# Each one is skipped when it is already built; pass --rebuild to build them again.
#
# Usage: scripts/ios/build-xcframeworks.sh [all|whisper] [--rebuild]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
set_name="all"
rebuild=false
for arg in "$@"; do
  case "${arg}" in
    all|whisper) set_name="${arg}" ;;
    --rebuild) rebuild=true ;;
    *) echo "usage: $0 [all|whisper] [--rebuild]" >&2; exit 2 ;;
  esac
done

modules=(runtime:GgmlRuntime whisper:KggmlWhisper)
[[ "${set_name}" == all ]] && modules+=(llama:KggmlLlama diffusion:KggmlDiffusion)
for entry in "${modules[@]}"; do
  module="${entry%%:*}" name="${entry##*:}"
  if [[ "${rebuild}" == false && -d "build/native-ios/${name}.xcframework" ]]; then
    echo "${name}.xcframework: already built (--rebuild to build it again)"
    continue
  fi
  "./kggml-${module}/native/scripts/build-ios-xcframework.sh"
done
