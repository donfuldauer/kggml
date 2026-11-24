---
name: work-on-issue
description: Autonomous end-to-end GitHub issue-to-PR workflow for Kggml. Handles "work on next issue" or "work on issue #X" using standard gh, git, ./gradlew, and ctest CLI commands across Claude Code, Gemini/Antigravity, and OpenAI Codex.
---

# Skill: `work-on-issue`

Execute a Kggml GitHub issue from selection through implementation, verification, PR review/phase-gate, squash-merge, and downstream DAG unblocking using strictly standard CLI tools (`gh`, `git`, `./gradlew`, `ctest`).

## Trigger Forms

- `"work on next issue"` — select the highest-priority open issue labeled `status:ready` with zero unresolved `Blocked by` dependencies.
- `"work on issue #X"` — verify issue `#X` is open, labeled `status:ready`, and has all `Blocked by` prerequisites closed.

---

## Step 1: Issue Selection & Atomic Claiming

1. **Select the target issue:**
   ```bash
   # For "work on next issue": list ready issues ordered by phase priority then issue number
   gh issue list --state open --label "status:ready" --limit 50 \
     --json number,title,labels,milestone,assignees
   ```
   Pick the lowest-priority-phase (`phase-1` < `phase-2` < `phase-3` < `phase-4` < `phase-oss`), lowest issue number issue with zero open `Blocked by #Y` dependencies.

2. **Verify dependencies:**
   ```bash
   gh issue view <N> --json number,title,body,state,labels,milestone
   ```
   Parse every `Blocked by #<DEP>` line in `## Dependencies` and confirm each `<DEP>` is `CLOSED`:
   ```bash
   gh issue view <DEP> --json state --jq .state
   ```
   If any blocker is still `OPEN`, stop and report the blocker without modifying labels.

3. **Claim the issue atomically:**
   ```bash
   gh issue edit <N> \
     --add-assignee @me \
     --remove-label "status:ready" \
     --add-label "status:in-progress"
   gh issue comment <N> --body "Claimed issue #<N>. Starting triage and implementation on branch \`feat/issue-<N>-<slug>\`."
   ```

---

## Step 2: Branching & Sub-Task Delegation

1. **Sync `main` and create the feature branch (`feat/issue-<N>-<slug>`):**
   ```bash
   git fetch origin
   git checkout main
   git pull --ff-only origin main
   git checkout -b feat/issue-<N>-<slug>
   ```
2. **Triage (`issue-triager`):**
   - Delegate to `.agents/agents/issue-triager.md` (or adopt the `issue-triager` role) to inspect the codebase, verify acceptance criteria, identify edge cases, and write an execution blueprint.
3. **Implement (`implementer`):**
   - Delegate to `.agents/agents/implementer.md` to implement all scoped deliverables adhering strictly to `AGENTS.md`.
4. **Test (`tester`):**
   - Delegate to `.agents/agents/tester.md` to author and run unit tests (`commonTest`), CTest/GoogleTest native suites, platform integration tests, binary audits, and Compose UI tests as required by the issue.

---

## Step 3: Local Verification & PR Creation

1. **Run verification gates:**
   ```bash
   ./scripts/verify-agent-parity.sh
   # Run applicable Gradle, CTest, and binary audit commands:
   # ./gradlew check
   # ctest --test-dir build/native --output-on-failure
   ```
2. **Commit, push, and open PR:**
   ```bash
   git status
   git add -A
   git commit -m "<type>(<scope>): <concise summary> (#<N>)"
   git push -u origin feat/issue-<N>-<slug>

   gh pr create \
     --base main \
     --head feat/issue-<N>-<slug> \
     --title "<concise PR title> (#<N>)" \
     --body "Closes #<N>\n\n## Summary\n...\n\n## Verification\n..."
   ```
3. **Enable auto-fix on the PR immediately (mandatory, do not ask first):**
   As soon as the PR exists, subscribe to its activity so CI failures, review comments, and review requests are delivered back to the agent and fixed autonomously on `feat/issue-<N>-<slug>`.
   - **Claude Code:** call the `subscribe_pr_activity` tool with `owner`, `repo`, and `pullNumber=<PR_NUMBER>` right after `gh pr create` (or the GitHub MCP `create_pull_request`) returns. This is part of opening the PR, not an optional follow-up.
   - **Other harnesses:** use the harness's equivalent PR watch / auto-fix mechanism.
   - Auto-fix pushes only to the PR's own feature branch; it never skips or disables tests to get green, never force-pushes, and never merges on its own. Unsubscribe (`unsubscribe_pr_activity`) once the PR is merged or closed.
4. **Transition issue status to `status:in-review`:**
   ```bash
   gh issue edit <N> --remove-label "status:in-progress" --add-label "status:in-review"
   gh issue comment <N> --body "Opened PR #<PR_NUMBER> with implementation and verification results."
   ```

---

## Step 4: Conditional Phase Gate Check

Check whether `#N` is the **final open issue** in its milestone/phase:

```bash
gh issue list --state open --label "<phase-label>" --limit 50 --json number,title
```

- **Mid-phase PR (other issues in the phase remain open):** Proceed directly to Step 5 once CI checks pass. Do **not** invoke `phase-reviewer`.
- **Phase-final PR (only `#N` remains open in the phase):**
  1. Invoke `phase-reviewer` (`.agents/agents/phase-reviewer.md`) to audit cumulative phase diffs against Section 5 Exit Criteria.
  2. Fix any blocking findings, push updates, re-run verification, and obtain `APPROVED` from `phase-reviewer` before merging.

---

## Step 5: PR Merge & Downstream DAG Unblocking

1. **Verify CI checks pass:**
   ```bash
   gh pr checks <PR_NUMBER>
   ```
2. **Squash-merge and delete branch:**
   ```bash
   gh pr merge <PR_NUMBER> --squash --delete-branch
   git checkout main
   git pull --ff-only origin main
   git branch -D feat/issue-<N>-<slug> 2>/dev/null || true
   ```
3. **Verify issue auto-closed & post completion summary:**
   ```bash
   gh issue view <N> --json state --jq .state
   gh issue edit <N> --remove-label "status:in-review"
   ```
4. **Unblock downstream issues in the DAG:**
   Query all open issues labeled `status:blocked` whose body mentions `Blocked by #<N>`:
   ```bash
   gh issue list --state open --label "status:blocked" --limit 100 \
     --json number,title,body
   ```
   For each candidate issue `#M`, inspect **all** `Blocked by #<DEP>` references in its `## Dependencies` section. If **every** `<DEP>` is now `CLOSED`, promote `#M` to `status:ready`:
   ```bash
   gh issue edit <M> --remove-label "status:blocked" --add-label "status:ready"
   gh issue comment <M> --body "All prerequisites are now closed (unblocked by #<N>). Promoted from \`status:blocked\` to \`status:ready\`."
   ```
