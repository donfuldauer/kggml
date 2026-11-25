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

## Phase 3

- [ ] [#12] Live dictation in `:samples:kggml-showcase` shows partial hypotheses with < 400 ms lag and commits
  punctuated, timestamped segments across >= 3 minutes of continuous speech, writing no WAV files — procedure in
  `samples/kggml-showcase/README.md` ("Live dictation latency and multi-minute continuity").
- [ ] [#12] On a physical iPhone, the Whisper Studio background stress toggle shows `CPU_ONLY` transcribing in the
  background while Metal is cut off, and the default `CPU_ONLY` fallback keeps the app alive — procedure in
  `samples/kggml-showcase/README.md` ("iOS background stress: `CPU_ONLY` vs. Metal").
- [ ] [#12] CI re-run on macOS/Windows once Actions minutes reset (compiles the showcase iOS AVAudioEngine capture and
  runs iOS simulator tests) — https://github.com/donfuldauer/kggml/pull/42

## Phase 4

- [ ] [#13] CI re-run on macOS/Windows once Actions minutes reset (macOS arm64 + `KggmlLlama.xcframework` iOS device /
  simulator build and Mach-O audit, Windows MSVC `kggml-llama` build + CTest) — https://github.com/donfuldauer/kggml/pull/44
- [ ] [#14] `LlamaEngine` with a real instruct / embedding GGUF on a physical iPhone and Android device and on desktop
  (Metal / Vulkan / CPU offload, coherent streaming, prompt cancellation latency, JSON-schema output, session
  restore, embedding similarity) — procedure in `kggml-llama/README.md` ("Manual verification").
