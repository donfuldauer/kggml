#!/usr/bin/env bash
# Merged Kover coverage of every JVM-tested module (commonTest + jvmTest / desktopTest / integration tests, plus the
# Android local unit tests of the debug/release variants) -> build/reports/kover/{html,report.xml}, and prints the
# line / branch totals. With the host native builds present (build/native*, from build-host.sh) the native-backed
# tests and JNI bridges are covered too; pass -Pkggml.requireNative=true to make their absence an error.
# Optional floor: KGGML_MIN_LINE_COVERAGE=<percent> fails the run below it.
# Shared by ./scripts/ci-local.sh (step `coverage`) and CI. Works on macOS, Linux and Windows (Git Bash) hosts.
#
# Usage: scripts/ci/coverage.sh [extra Gradle args...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck source=python.sh
source "${REPO_ROOT}/scripts/ci/python.sh"
kggml_require_python 3.8

./gradlew --console=plain koverXmlReport koverHtmlReport \
  -x :kggml-runtime:buildNativeHost -x :kggml-whisper:buildNativeHost \
  -x :kggml-llama:buildNativeHost -x :kggml-diffusion:buildNativeHost "$@"

kggml_python - "build/reports/kover/report.xml" "${KGGML_MIN_LINE_COVERAGE:-0}" <<'PY'
import sys
import xml.etree.ElementTree as ET

report, floor = sys.argv[1], float(sys.argv[2])
root = ET.parse(report).getroot()
totals = {c.get("type"): (int(c.get("missed")), int(c.get("covered"))) for c in root.findall("counter")}

def pct(kind):
    missed, covered = totals.get(kind, (0, 0))
    return 100.0 * covered / max(1, missed + covered), covered, missed + covered

line, lc, lt = pct("LINE")
branch, bc, bt = pct("BRANCH")
print(f"Kover coverage: lines {line:.1f}% ({lc}/{lt}), branches {branch:.1f}% ({bc}/{bt})")
print("HTML report: build/reports/kover/html/index.html")
if line < floor:
    sys.exit(f"FAIL: line coverage {line:.1f}% is below KGGML_MIN_LINE_COVERAGE={floor}%")
PY
