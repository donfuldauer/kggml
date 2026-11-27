#!/usr/bin/env bash
# Reads and sets an issue's GitHub status (the issue page's Backlog / Todo / In progress / In review / Done /
# Not planned / Duplicate field), which drives the work-on-issue workflow: "work on next issue" only takes issues in
# Todo and ignores Backlog. `gh issue` has no flag for this field yet, so this uses the GraphQL API.
#
# Usage:
#   scripts/issue-status.sh get <issue-number>              prints the status name
#   scripts/issue-status.sh set <issue-number> "<status>"   e.g. "In progress"; prints the new status
#   scripts/issue-status.sh list "<status>"                 open issue numbers in that status, lowest first
# The repository is the one `gh repo view` resolves (override with GH_REPO=owner/name).
set -euo pipefail

usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
[[ $# -ge 2 ]] || usage 2

repo="${GH_REPO:-$(gh repo view --json nameWithOwner --jq .nameWithOwner)}"
owner="${repo%/*}"; name="${repo#*/}"

status_of() {
  gh api graphql -f query='query($o:String!,$n:String!,$i:Int!){repository(owner:$o,name:$n){issue(number:$i){status{name}}}}' \
    -f o="${owner}" -f n="${name}" -F i="$1" --jq '.data.repository.issue.status.name // ""'
}

case "$1" in
  get)
    status_of "$2" ;;
  set)
    [[ $# -eq 3 ]] || usage 2
    issue_id="$(gh issue view "$2" --repo "${repo}" --json id --jq .id)"
    status_id="$(gh api graphql -f query='query($o:String!,$n:String!){repository(owner:$o,name:$n){planningStatuses(first:50){nodes{id name}}}}' \
      -f o="${owner}" -f n="${name}" --jq ".data.repository.planningStatuses.nodes[] | select(.name == \"$3\") | .id")"
    [[ -n "${status_id}" ]] || { echo "issue-status: no status named '$3' in ${repo}" >&2; exit 1; }
    gh api graphql -f query='mutation($i:ID!,$s:ID!){setIssueStatus(input:{issueId:$i,statusId:$s}){clientMutationId}}' \
      -f i="${issue_id}" -f s="${status_id}" >/dev/null
    status_of "$2" ;;
  list)
    gh issue list --repo "${repo}" --state open --search "status:\"$2\"" --limit 200 --json number --jq '.[].number' | sort -n ;;
  -h|--help|help) usage ;;
  *) usage 2 ;;
esac
