#!/usr/bin/env bash
# Binary symbol leakage audit of :samples:whisper-only-minimal (AGENTS.md §2.1) on its shipped artifacts, via
# scripts/audit-binary-symbols.sh. Shared by .github/workflows/ci.yml and ./scripts/ci-local.sh.
#
#   android   Self-tests the audit script, assembles the release APK with the native libraries bundled
#             (-Pkggml.bundleNative=true; needs ANDROID_NDK_HOME) and audits every ABI in it.
#   desktop   Audits the JVM desktop app: its jar plus resolved runtime classpath (no llama / diffusion / GPU add-on
#             classes or native libraries; whisper present), and the host native libraries it loads
#             (build/native/lib + build/native-whisper/lib, from the build-host.sh scripts) as one set: ELF on
#             Linux, Mach-O on macOS (Windows DLLs: jars only).
#   ios       (macOS) Audits WhisperOnlyMinimal.framework (debug, iosArm64 + iosSimulatorArm64; link them first
#             with :samples:whisper-only-minimal:linkDebugFrameworkIos*) together with the matching slices of
#             GgmlRuntime.xcframework and KggmlWhisper.xcframework.
#
# Extra arguments are passed to Gradle (android mode), e.g. -x :kggml-runtime:buildNativeAndroid.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
AUDIT="${REPO_ROOT}/scripts/audit-binary-symbols.sh"

mode="${1:-}"
shift || true
case "${mode}" in
  android)
    ./scripts/tests/test-audit-binary-symbols.sh
    ./gradlew :samples:whisper-only-minimal:assembleRelease -Pkggml.bundleNative=true --console=plain "$@"
    "${AUDIT}" apk samples/whisper-only-minimal/build/outputs/apk/release/whisper-only-minimal-release-unsigned.apk
    ;;
  desktop)
    # --console=plain unless the caller passed its own (Gradle rejects a repeated --console, as CI's
    # NATIVE_GRADLE_ARGS carries one).
    gradle_args=("$@")
    [[ " $* " == *" --console"* ]] || gradle_args+=(--console=plain)
    ./gradlew :samples:whisper-only-minimal:desktopDistribution "${gradle_args[@]}"
    "${AUDIT}" jars samples/whisper-only-minimal/build/desktop-dist/lib
    host_os="$(uname -s)"
    if [[ "${host_os}" == Linux || "${host_os}" == Darwin ]]; then
      for dir in build/native/lib build/native-whisper/lib; do
        [[ -d "${dir}" ]] || { echo "missing ${dir}: run the kggml-runtime / kggml-whisper build-host.sh scripts" >&2; exit 1; }
      done
    fi
    case "${host_os}" in
      Linux) "${AUDIT}" elf build/native/lib build/native-whisper/lib ;;
      Darwin)
        # The real Mach-O dylibs (versioned files, not their symlinks), audited as one set like the iOS frameworks.
        dylibs=()
        while IFS= read -r f; do dylibs+=("${f}"); done \
          < <(find build/native/lib build/native-whisper/lib -maxdepth 1 -type f -name '*.dylib' | sort)
        [[ ${#dylibs[@]} -gt 0 ]] || { echo "no host dylibs in build/native/lib or build/native-whisper/lib" >&2; exit 1; }
        "${AUDIT}" ios "${dylibs[@]}" ;;
      *) echo "desktop native libraries: audited on Linux (ELF) and macOS (Mach-O) only; jars audited above" ;;
    esac
    ;;
  ios)
    native="${REPO_ROOT}/build/native-ios"
    for pair in iosArm64:ios-arm64 iosSimulatorArm64:ios-arm64-simulator; do
      target="${pair%%:*}" slice="${pair##*:}"
      echo "==> ${target} (${slice})"
      "${AUDIT}" ios \
        "samples/whisper-only-minimal/build/bin/${target}/debugFramework/WhisperOnlyMinimal.framework" \
        "${native}/GgmlRuntime.xcframework/${slice}/GgmlRuntime.framework" \
        "${native}/KggmlWhisper.xcframework/${slice}/KggmlWhisper.framework"
    done
    ;;
  *)
    echo "usage: $0 android [gradle args...] | desktop [gradle args...] | ios" >&2
    exit 2
    ;;
esac
