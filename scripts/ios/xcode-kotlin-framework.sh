#!/usr/bin/env bash
# "Compile Kotlin framework" build phase of the samples' Xcode host apps (samples/*/iosApp). Builds the sample's static
# Kotlin framework for the configuration / SDK / architectures Xcode is building (Kotlin's
# embedAndSignAppleFrameworkForXcode, output in samples/<sample>/build/xcode-frameworks/$CONFIGURATION/$SDK_NAME).
#
# Usage (Xcode Run Script phase): "$SRCROOT/../../../scripts/ios/xcode-kotlin-framework.sh" :samples:<sample> <xcframework>...
#   <xcframework>: the native frameworks the app embeds (e.g. GgmlRuntime KggmlWhisper); only checked, built by
#   scripts/ios/build-xcframeworks.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
project="${1:?usage: xcode-kotlin-framework.sh :samples:<sample> <xcframework>...}"
shift

missing=()
for name in "$@"; do
  [[ -d "${REPO_ROOT}/build/native-ios/${name}.xcframework" ]] || missing+=("${name}")
done
if ((${#missing[@]})); then
  echo "error: missing ${missing[*]/%/.xcframework} in build/native-ios: run scripts/ios/build-xcframeworks.sh first"
  exit 1
fi

# Xcode runs build phases with a minimal environment (no shell profile): find JDK 17 the way a shell would.
if [[ -z "${JAVA_HOME:-}" || ! -x "${JAVA_HOME}/bin/java" ]]; then
  for candidate in \
    "$(/usr/libexec/java_home -v 17 2>/dev/null || true)" \
    /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
    /usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home; do
    if [[ -n "${candidate}" && -x "${candidate}/bin/java" ]]; then export JAVA_HOME="${candidate}"; break; fi
  done
fi
[[ -x "${JAVA_HOME:-}/bin/java" ]] || { echo "error: JDK 17 not found: set JAVA_HOME (e.g. brew install openjdk@17)"; exit 1; }

cd "${REPO_ROOT}"
exec ./gradlew --console=plain "${project}:embedAndSignAppleFrameworkForXcode"
