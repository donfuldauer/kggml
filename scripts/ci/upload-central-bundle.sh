#!/usr/bin/env bash
# Uploads a bundle made by scripts/ci/stage-release.sh to the Maven Central Publisher Portal and waits until Central
# has validated it. Used only by .github/workflows/release.yml, never by PR CI.
#
# Credentials: a Central Portal user token in MAVEN_CENTRAL_USERNAME / MAVEN_CENTRAL_PASSWORD (repository secrets).
#
# Usage: scripts/ci/upload-central-bundle.sh <bundle.zip> <deployment-name> [USER_MANAGED|AUTOMATIC]
#   USER_MANAGED (default): Central validates the deployment and the owner presses "Publish" in the portal
#                           (https://central.sonatype.com/publishing/deployments), which is the last manual gate.
#   AUTOMATIC:              Central publishes as soon as validation passes. Publishing to Central is permanent.
# Exit status: 0 validated (USER_MANAGED) or publishing/published (AUTOMATIC), 1 Central rejected the bundle or
# the request failed, 2 Central had not finished after the deadline (30 min).
set -euo pipefail

bundle="${1:?usage: upload-central-bundle.sh <bundle.zip> <deployment-name> [USER_MANAGED|AUTOMATIC]}"
name="${2:?deployment name required}"
publishing_type="${3:-USER_MANAGED}"
api="${KGGML_CENTRAL_API:-https://central.sonatype.com/api/v1/publisher}"

case "${publishing_type}" in USER_MANAGED|AUTOMATIC) ;; *) echo "FAIL: unknown publishing type ${publishing_type}" >&2; exit 1 ;; esac
[[ -f "${bundle}" ]] || { echo "FAIL: bundle ${bundle} not found" >&2; exit 1; }
: "${MAVEN_CENTRAL_USERNAME:?MAVEN_CENTRAL_USERNAME (Central Portal user token name) is not set}"
: "${MAVEN_CENTRAL_PASSWORD:?MAVEN_CENTRAL_PASSWORD (Central Portal user token password) is not set}"

token="$(printf '%s:%s' "${MAVEN_CENTRAL_USERNAME}" "${MAVEN_CENTRAL_PASSWORD}" | base64 | tr -d '\n')"
# The header goes through a file so the token never appears in a process listing.
header_file="$(mktemp)"
trap 'rm -f "${header_file}"' EXIT
printf 'Authorization: Bearer %s\n' "${token}" >"${header_file}"

echo "==> Uploading ${bundle} as '${name}' (${publishing_type})"
deployment_id="$(curl --fail-with-body --silent --show-error --retry 3 --retry-all-errors \
  -H @"${header_file}" \
  --form-string "name=${name}" \
  --form "bundle=@${bundle};type=application/zip" \
  "${api}/upload?publishingType=${publishing_type}")"
deployment_id="$(tr -d '[:space:]"' <<<"${deployment_id}")"
[[ "${deployment_id}" =~ ^[A-Za-z0-9-]+$ ]] || { echo "FAIL: unexpected upload response: ${deployment_id}" >&2; exit 1; }
echo "Deployment id: ${deployment_id}"
[[ -n "${GITHUB_OUTPUT:-}" ]] && echo "deployment_id=${deployment_id}" >>"${GITHUB_OUTPUT}"

deadline=$(( $(date +%s) + 30 * 60 ))
while :; do
  status_json="$(curl --fail-with-body --silent --show-error --retry 3 --retry-all-errors -X POST \
    -H @"${header_file}" "${api}/status?id=${deployment_id}")"
  state="$(python3 -I -c 'import json,sys; print(json.load(sys.stdin).get("deploymentState",""))' <<<"${status_json}")"
  echo "$(date -u +%H:%M:%S) deployment ${deployment_id}: ${state}"
  case "${state}" in
    VALIDATED)
      if [[ "${publishing_type}" == USER_MANAGED ]]; then
        echo "Validated. Publish it from https://central.sonatype.com/publishing/deployments"
        exit 0
      fi ;;
    PUBLISHING|PUBLISHED) echo "Central is publishing deployment ${deployment_id}."; exit 0 ;;
    FAILED)
      echo "FAIL: Central rejected the deployment:" >&2
      python3 -I -c 'import json,sys; print(json.dumps(json.load(sys.stdin).get("errors", {}), indent=2))' <<<"${status_json}" >&2
      exit 1 ;;
  esac
  if (( $(date +%s) > deadline )); then
    echo "FAIL: deployment ${deployment_id} still ${state} after 30 minutes" >&2
    exit 2
  fi
  sleep 20
done
