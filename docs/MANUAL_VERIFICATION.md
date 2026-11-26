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
- [ ] [#15] VLM image prompts with a real vision GGUF + `mmproj` on a physical iPhone and Android device and on desktop
  (camera JPEG and `Rgb` bitmap described correctly, GPU / CPU projector, abort mid-stream, no files in cache / temp
  directories) — procedure in `kggml-llama/README.md` ("Manual verification", step 7).
- [ ] [#14] CI re-run on Windows (billing / spending-limit refusal: MSVC build of the new `kggml_llama_jni` bridge +
  `:kggml-llama:jvmTest`) — https://github.com/donfuldauer/kggml/pull/46
- [ ] [#16] `libkggml_diffusion` txt2img with a real SD 1.5-class GGUF through the C ABI on desktop (CPU, plus Metal on
  macOS / Vulkan where available): steps reported in order, a coherent image, abort within one step and the context
  reusable afterwards — procedure in `kggml-diffusion/native/README.md` ("Manual verification").
- [ ] [#16] CI re-run on Windows (billing / spending-limit refusal, job failed with no logs: MSVC `kggml-diffusion` build
  incl. `/EHs /EHc-`, `GGML_MAX_NAME=128` runtime/whisper/llama rebuild + CTest) — https://github.com/donfuldauer/kggml/pull/49
- [ ] [#17] `DiffusionEngine` with a real SD 1.5-class model on a physical iPhone and Android device and on desktop
  (Metal / Vulkan / CPU, progress `0..N` in order, a coherent 512x512 image, same seed = same image, cancellation within
  one step and the engine reusable afterwards, `CPU_ONLY` in the iOS background, stable memory across 10 generations) —
  procedure in `kggml-diffusion/README.md` ("Manual verification").
- [ ] [#17] CI re-run on Windows (billing / spending-limit refusal, job failed with no logs: MSVC `kggml_diffusion_jni`
  build + `:kggml-diffusion:jvmTest` against the native libs) — https://github.com/donfuldauer/kggml/pull/51
- [ ] [#18] Model manager on physical Android/iOS devices (real GGUF download resumed after airplane mode and app
  kill, SHA-256 mismatch leaves no `.part`, `send-trim-memory` / iOS memory warning closes idle resident engines,
  peak-RAM estimate within ~±30% of measured RSS) — procedure in `kggml-model-manager/README.md` ("Manual verification").
- [ ] [#19] Showcase Model Manager real downloads on a physical Android phone and iPhone (progress, pause / resume, resume
  after airplane mode and app kill, SHA-256 shown and matching Hugging Face, metadata + peak-RAM estimate, "Use in
  Chat") — procedure in `samples/kggml-showcase/README.md` ("Model Manager: real downloads on devices").
- [ ] [#19] Showcase OS low-memory eviction on devices (`adb shell am send-trim-memory … RUNNING_CRITICAL` /
  `RUNNING_MODERATE`, iOS Simulate Memory Warning and device memory pressure: idle engines closed and RSS released,
  leased engine survives, reload on next use) — procedure in `samples/kggml-showcase/README.md` ("Model Manager: OS
  low-memory eviction on devices").
- [ ] [#19] Showcase LLM Chat / structured output / embeddings / VLM with real GGUF models on desktop, Android and iPhone
  (streaming, cancel, thread / GPU-layer reload, KV snapshot restore, JSON-schema output, camera + library images with
  no temp files) — procedure in `samples/kggml-showcase/README.md` ("LLM Chat, structured output and vision on
  hardware").
- [ ] [#19] Showcase Diffusion Image Studio with a real SD 1.5 model on desktop, Android and iPhone (progress, same seed =
  same image, cancel within one step, PNG export opens in an image viewer) — procedure in
  `samples/kggml-showcase/README.md` ("Diffusion Image Studio on hardware").
- [ ] [#20] Whisper live dictation and LLM Chat streaming at the same time in `:samples:kggml-showcase` on a physical
  Android phone, an iPhone and desktop (both keep producing output, no crash or OOM, one mapping per `libggml*` library
  on Android) — procedure in `integration-tests/README.md` ("Whisper and Llama together on devices").
- [ ] [#20] CI re-run on Windows (billing / spending-limit refusal, job failed with no logs: MSVC runtime / whisper /
  llama / diffusion build + CTest and root `jvmTest` incl. `:integration-tests` dual-engine coexistence) —
  https://github.com/donfuldauer/kggml/pull/54
- [ ] [#22] External links in the Markdown docs once the repository is public (GitHub issue-template, advisory and
  upstream links answer 404/403 to unauthenticated requests while it is private, and CI has no network rule for
  them): `./scripts/ci/verify-docs.sh --online --only links` reports no failures — procedure in `CONTRIBUTING.md`
  ("Documentation").
- [ ] [#22] README quickstart on real hardware: the `README.md` Whisper, Llama, Diffusion and model-manager snippets
  (compiled by `:docs-snippets`) produce sensible output with real models on desktop, a physical Android phone and an
  iPhone, and the iOS background guidance (`BackendConfig.CPU` / `observeAppLifecycle`) keeps transcription running
  in the background — the same procedures as the module READMEs' "Manual verification" sections.

## Phase OSS

- [ ] [#63] Owner, before the repository is made public: unregister the Mac, Windows and Linux self-hosted runners
  (added back in #62), delete `MAC_RUNNER`, `WIN_RUNNER` and `LINUX_RUNNER`, and merge the PR that removes the
  self-hosted routes; `scripts/ci/verify-hosted-runners.sh` fails while public until then — procedure in
  `docs/OSS_LAUNCH.md` ("Owner steps before the visibility flip", step 1).
- [ ] [#23] Owner: Central Portal namespace `com.donfuldauer` verified, release secrets added, public signing key on a
  keyserver, then a `release.yml` dry-run dispatch from `main` is green and its bundle validates in the portal —
  procedure in `docs/RELEASING.md` ("Before the first release" and "Dry run").
- [ ] [#23] Full `ci.yml` matrix (Ubuntu, macOS, Windows) green on GitHub-hosted runners from `main` after merge
  (`Actions → CI → Run workflow`) — procedure in `docs/RELEASING.md` ("Dry run").
- [ ] [#23] CI re-run on macOS/Windows once Actions minutes reset (billing / spending-limit refusal, every hosted job
  failed in ~2 s with no logs: macOS release dry-run with the iOS klibs, `release.yml` iOS/xcframework path, Windows
  MSVC build + `jvmTest`) — https://github.com/donfuldauer/kggml/pull/57
- [ ] [#24] Owner, before the visibility flip: no runner registered to the repository (`gh api repos/donfuldauer/kggml/actions/runners
  --jq .total_count` prints `0`, `gh variable list` shows no runner variable) — procedure in `docs/OSS_LAUNCH.md`
  ("Owner steps before the visibility flip", step 1).
- [ ] [#24] Owner: GitHub secret scanning and push protection enabled (Settings → Code security), and no secret
  scanning alerts after the flip — procedure in `docs/OSS_LAUNCH.md` ("Owner steps before the visibility flip",
  step 2).
- [ ] [#24] Owner decision: keep or rewrite the personal commit e-mail on 12 early/merge commits
  (`./scripts/ci/audit-history.sh` lists them without the address) — procedure in `docs/OSS_LAUNCH.md` ("Commit
  e-mail addresses").
- [ ] [#24] Owner: `./scripts/ci/audit-history.sh` passes on a fresh full clone right before the visibility flip, then
  flip to public — procedure in `docs/OSS_LAUNCH.md` ("Owner steps before the visibility flip", steps 4–5).
- [ ] [#24] CI re-run once Actions minutes reset (billing / spending-limit refusal: every hosted job, including
  repo-hygiene's `secret-scan` full-history audit with the pinned gitleaks download, the Ubuntu job and the labelled
  macOS/Windows jobs, failed in 1–3 s with no logs) — https://github.com/donfuldauer/kggml/pull/58
