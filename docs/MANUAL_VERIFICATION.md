# Deferred manual and physical verification

Checks that need a person or a physical device. Agent sessions never wait for these: each one is
documented and listed here by the PR that introduces it, and the PR merges on its automated gates
(AGENTS.md Section 3, "Deferred Manual & Physical Verification"). The owner works through this
list once, after every phase is complete.

Format: `- [ ] [#<issue>] <check> — <procedure link>`. Tick an item (with the date and result)
when it has been run; a failed check becomes a new issue.

## Phase 2

- [ ] [#9] On a physical iPhone, `ComputeBackend.CPU_ONLY` completes Whisper transcription while
  the app is backgrounded, with no Metal `accessRevoked` abort — procedure in
  `samples/whisper-only-minimal/README.md` ("Physical iOS background `CPU_ONLY` verification").
