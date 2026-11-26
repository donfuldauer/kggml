#!/usr/bin/env bash
# Stages a signed Maven Central release of every Kggml coordinate, audits it, and zips it as a Central Portal
# upload bundle. Shared by .github/workflows/release.yml (real release), the release dry-run steps of ci.yml
# (Ubuntu: JVM + Android + metadata; macOS: every target incl. iOS) and ./scripts/ci-local.sh (`release`).
#
#   1. `./gradlew publishAllPublicationsToKggmlStagingRepository -Pkggml.release=true` into a clean
#      build/kggml-staging-repo (the `kggml.publishing` convention plugin signs every publication and adds
#      the javadoc jars Central requires).
#   2. scripts/ci/verify_release.py: POM metadata, Gradle .module files, BOM constraints, sources/javadoc jars,
#      md5/sha1/sha256/sha512 checksums, and every .asc verified with gpg against the signing key.
#   3. build/kggml-central-bundle.zip: the staged files without maven-metadata*.xml and checksums of the .asc files
#      (Central neither needs nor wants them).
#
# Signing key: ORG_GRADLE_PROJECT_signingInMemoryKey (ASCII-armored secret key), optional
# ORG_GRADLE_PROJECT_signingInMemoryKeyPassword / ORG_GRADLE_PROJECT_signingInMemoryKeyId. With --dry-run and no
# key, a throwaway key is generated for this run only, so the signing path is exercised without any secret.
#
# Usage: scripts/ci/stage-release.sh [--dry-run] [--skip-targets <list>] [--expect-version <v>] [-- <gradle args>]
#   --dry-run          allow an ephemeral signing key when none is configured
#   --skip-targets     comma-separated targets the host cannot build, skipped by the audit (e.g. `ios` on Linux)
#   --expect-version   fail unless VERSION_NAME equals this (release.yml passes the tag without its `v`)
#   <gradle args>      extra Gradle arguments (e.g. -Pkggml.bundleNative=true)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

dry_run=0
skip_targets=""
expect_version=""
gradle_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry_run=1 ;;
    --skip-targets) skip_targets="${2:?--skip-targets needs a list}"; shift ;;
    --expect-version) expect_version="${2:?--expect-version needs a version}"; shift ;;
    --) shift; gradle_args=("$@"); break ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

version="$(sed -n 's/^VERSION_NAME=//p' gradle.properties | tail -n 1)"
group="$(sed -n 's/^GROUP=//p' gradle.properties | tail -n 1)"
[[ -n "${version}" && -n "${group}" ]] || { echo "FAIL: GROUP / VERSION_NAME missing from gradle.properties" >&2; exit 1; }
if [[ -n "${expect_version}" && "${expect_version}" != "${version}" ]]; then
  echo "FAIL: release tag version ${expect_version} != VERSION_NAME ${version} in gradle.properties" >&2
  exit 1
fi
if [[ "${version}" == *-SNAPSHOT ]]; then
  echo "FAIL: ${version} is a SNAPSHOT; Maven Central releases need a fixed version" >&2
  exit 1
fi

GNUPGHOME_DIR="$(mktemp -d)"
chmod 700 "${GNUPGHOME_DIR}"
trap 'gpgconf --homedir "${GNUPGHOME_DIR}" --kill all >/dev/null 2>&1 || true; rm -rf "${GNUPGHOME_DIR}"' EXIT
gpg_() { gpg --homedir "${GNUPGHOME_DIR}" --batch --quiet "$@"; }

if [[ -z "${ORG_GRADLE_PROJECT_signingInMemoryKey:-}" ]]; then
  if [[ ${dry_run} -eq 0 ]]; then
    echo "FAIL: no signing key (ORG_GRADLE_PROJECT_signingInMemoryKey); pass --dry-run for an ephemeral key" >&2
    exit 1
  fi
  echo "==> Dry run: generating an ephemeral signing key (never leaves this run)"
  gpg_ --passphrase '' --pinentry-mode loopback \
    --quick-generate-key "Kggml release dry run <dry-run@kggml.invalid>" rsa3072 sign 1d
  ORG_GRADLE_PROJECT_signingInMemoryKey="$(gpg_ --armor --pinentry-mode loopback --passphrase '' --export-secret-keys)"
  ORG_GRADLE_PROJECT_signingInMemoryKeyPassword=""
  export ORG_GRADLE_PROJECT_signingInMemoryKey ORG_GRADLE_PROJECT_signingInMemoryKeyPassword
else
  echo "==> Using the configured signing key"
  # Importing a secret key needs no passphrase; it gives gpg the public key to verify signatures with.
  printf '%s\n' "${ORG_GRADLE_PROJECT_signingInMemoryKey}" | gpg_ --import
fi

STAGING="${REPO_ROOT}/build/kggml-staging-repo"
BUNDLE="${REPO_ROOT}/build/kggml-central-bundle.zip"
rm -rf "${STAGING}" "${BUNDLE}"

echo "==> Publishing every coordinate to ${STAGING#"${REPO_ROOT}/"} (signed)"
# Plain console output unless the caller chose one: Gradle rejects a second --console (ci.yml passes
# $NATIVE_GRADLE_ARGS, which already has --console=plain; issue #69).
console_args=(--console=plain)
for arg in ${gradle_args[@]+"${gradle_args[@]}"}; do
  [[ "${arg}" == --console || "${arg}" == --console=* ]] && console_args=()
done
./gradlew publishAllPublicationsToKggmlStagingRepository -Pkggml.release=true \
  ${console_args[@]+"${console_args[@]}"} ${gradle_args[@]+"${gradle_args[@]}"}

echo "==> Auditing the staged release"
python3 -I "${REPO_ROOT}/scripts/ci/verify_release.py" "${STAGING}" --version "${version}" --group "${group}" \
  --skip-targets "${skip_targets}" --gnupg-home "${GNUPGHOME_DIR}"

echo "==> Bundling for the Central Portal"
(cd "${STAGING}" && find . -type f ! -name 'maven-metadata*' ! -name '*.asc.*' | LC_ALL=C sort | sed 's|^\./||' | zip -q -X "${BUNDLE}" -@)
echo "Bundle: ${BUNDLE#"${REPO_ROOT}/"} ($(unzip -l "${BUNDLE}" | tail -n 1 | awk '{print $2}') files)"
