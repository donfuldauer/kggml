# Git history audit

Every pull request audits the repository's whole git history, not just the current tree, so that nothing that should
never be published (a secret, a credential file, a large binary, a developer's machine path) can sit in an old commit.

[`scripts/ci/audit-history.sh`](../scripts/ci/audit-history.sh) scans every commit reachable from any branch, tag or
fetched pull-request head. It fails if any commit ever contained:

| Check | What fails |
| --- | --- |
| Secrets | anything `gitleaks git --log-opts=--all` reports, using [`.gitleaks.toml`](../.gitleaks.toml) |
| Blob size | any blob over 1 MiB (`KGGML_HISTORY_MAX_BLOB_BYTES`) |
| Credential and local files | `.pem`, `.key`, `.p12`, `.pfx`, `.jks`, `.keystore`, `.gpg`, `.p8`, `.env` (except `.env.example`), `local.properties`, `.DS_Store`, SSH keys, `credentials.json`, `google-services.json`, `GoogleService-Info.plist`, even if a later commit deleted them |
| Binaries | a model, audio, image, archive or compiled file whose path is not in [`NOTICE`](../NOTICE) |
| Private keys | a `BEGIN … PRIVATE KEY` or PGP private key block in any file or commit message |
| Developer paths | a `/Users/<name>/`, `/home/<name>/`, root home directory (`/root`) or `C:\Users\<name>\` path in any file or commit message, except the placeholder names `someone`, `example`, `runner` and `username` (`KGGML_HISTORY_PATH_ALLOW`) |

It also reports, without failing, commits whose author or committer e-mail is not a GitHub or Anthropic no-reply
address. The address itself is never printed (see [Commit e-mail addresses](#commit-e-mail-addresses)).

It runs in the `secret-scan` job of [`repo-hygiene.yml`](../.github/workflows/repo-hygiene.yml) on every PR, over
every branch plus the PR's own head. Pull-request heads stay on GitHub read-only, so one bad fork PR must not fail
every later PR: a manual dispatch of `repo-hygiene.yml` audits every pull-request head too. It also runs in
`./scripts/ci-local.sh` (step `gitleaks`), and its tests are in
[`scripts/tests/test-audit-history.sh`](../scripts/tests/test-audit-history.sh). To run it by hand on a fresh full
clone:

```bash
git clone https://github.com/donfuldauer/kggml && cd kggml
git fetch origin '+refs/heads/*:refs/remotes/origin/*' '+refs/pull/*/head:refs/remotes/pull/*'
./scripts/ci/audit-history.sh
```

## Commit e-mail addresses

GitHub shows the author and committer e-mail of every commit. To keep a personal address out of new commits, turn on
Settings → Emails → "Keep my email addresses private" and "Block command line pushes that expose my email" on GitHub,
and commit with the `<id>+<user>@users.noreply.github.com` address. `./scripts/ci/audit-history.sh
--strict-identity` fails on any commit whose author or committer is not a no-reply address.

## Remediating a finding

If the audit reports a real finding, only the owner rewrites history. In a public repository a rewrite cannot
recall what was already cloned or cached, which is why rotating comes first:

1. **Rotate first.** A leaked secret stays compromised even after it leaves history. Revoke or rotate it at the
   provider before anything else.
2. Merge or close every open pull request, and stop scheduled agent sessions.
3. In a fresh mirror clone, remove the finding. Install `git filter-repo` with `pip install git-filter-repo`.

   ```bash
   git clone --mirror https://github.com/donfuldauer/kggml kggml-rewrite.git && cd kggml-rewrite.git
   git filter-repo --invert-paths --path <file-to-drop>             # a file
   git filter-repo --replace-text <expressions-file>                # a string, e.g. 'literal==>REDACTED'
   git filter-repo --strip-blobs-bigger-than 1M                     # oversized blobs
   ```

4. Run `./scripts/ci/audit-history.sh --repo <rewritten clone>` until it passes, then push the result with
   `git push --force --mirror`. GitHub keeps pull-request refs (`refs/pull/*`) read-only. If the finding is in one,
   ask GitHub Support to purge the cached views and dangling objects.
5. Every collaborator re-clones. Pinned SHAs in issues and PR bodies no longer resolve.
