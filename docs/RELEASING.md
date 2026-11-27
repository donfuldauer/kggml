# Releasing Kggml to Maven Central

Kggml is published to Maven Central under `com.donfuldauer` by [`release.yml`](../.github/workflows/release.yml).
No version has been published yet; the first release is planned as 0.1.0.
One release publishes all nine coordinates at the `VERSION_NAME` in [`gradle.properties`](../gradle.properties):

| Coordinate | Contents |
| --- | --- |
| `com.donfuldauer:kggml-bom` | BOM (`pom` packaging) constraining the eight libraries below |
| `com.donfuldauer:kggml-core-api` | Kotlin Multiplatform library: root + `-android`, `-jvm`, `-iosarm64`, `-iossimulatorarm64`, `-macosarm64` |
| `com.donfuldauer:kggml-runtime` | same (with `-macosarm64`); the `-jvm` jar carries the runtime for `linux-x86_64`, `linux-aarch64`, `windows-x86_64`, `windows-aarch64`, `macos-aarch64`; the AAR carries `arm64-v8a` and `x86_64` |
| `com.donfuldauer:kggml-runtime-vulkan` | root + `-android`, `-jvm`, `-iosarm64`, `-iossimulatorarm64`; optional GPU add-on: the AAR carries `libggml-vulkan.so` (`arm64-v8a`, `x86_64`), the `-jvm` jar the `linux-x86_64` Vulkan module (no Windows module yet); no-op on iOS |
| `com.donfuldauer:kggml-runtime-opencl` | root + `-android`, `-jvm`, `-iosarm64`, `-iossimulatorarm64`; optional Android GPU add-on: the AAR carries `libggml-opencl.so` (`arm64-v8a`, `x86_64`); no-op on JVM and iOS |
| `com.donfuldauer:kggml-whisper` | root + `-android`, `-jvm`, `-iosarm64`, `-iossimulatorarm64`; the AAR carries the whisper JNI libraries |
| `com.donfuldauer:kggml-llama` | same; the AAR carries the llama JNI libraries |
| `com.donfuldauer:kggml-diffusion` | same; the AAR carries the diffusion JNI libraries |
| `com.donfuldauer:kggml-model-manager` | same (pure Kotlin) |

Every component has a POM (MIT license, SCM, developer), Gradle module metadata (`.module`), a `-sources.jar`, a
`-javadoc.jar` (a pointer to the KDoc in the sources jar), `.md5`/`.sha1`/`.sha256`/`.sha512` checksums and a PGP
signature (`.asc`) on every file. The iOS xcframeworks (`GgmlRuntime`, `KggmlWhisper`, `KggmlLlama`,
`KggmlDiffusion`) are attached to a draft GitHub release with a `SHA256SUMS` file.

## How it works

1. **`desktop-natives`** (`ubuntu-latest`, `windows-latest`): builds the release runtime for `linux-x86_64` and
   `windows-x86_64`, cross-compiles it for `linux-aarch64` (GNU aarch64 toolchain) and `windows-aarch64` (LLVM clang,
   MSVC ABI), stages each as JVM resources and checks every library's machine type
   ([`check_native_arch.py`](../scripts/ci/check_native_arch.py)). The `linux-x86_64` job also builds the desktop
   Vulkan add-on module for the `kggml-runtime-vulkan` jar.
2. **`publish`** (`macos-latest`, the only host that builds every Kotlin target): builds the iOS xcframeworks, then
   runs [`scripts/ci/stage-release.sh`](../scripts/ci/stage-release.sh), which
   - publishes every publication with `-Pkggml.release=true -Pkggml.bundleNative=true` into
     `build/kggml-staging-repo`, signed by the `kggml.publishing` convention plugin
     ([`KggmlPublishingPlugin.kt`](../build-logic/convention/src/main/kotlin/com/donfuldauer/kggml/buildlogic/KggmlPublishingPlugin.kt));
   - audits it with [`scripts/ci/verify_release.py`](../scripts/ci/verify_release.py): POM metadata, `.module`
     files and the platform components they point at, BOM constraints, sources/javadoc jars, checksums, and every
     signature verified with `gpg`;
   - zips `build/kggml-central-bundle.zip` for the Central Publisher Portal.
3. On a real release, [`scripts/ci/upload-central-bundle.sh`](../scripts/ci/upload-central-bundle.sh) uploads the
   bundle and waits until Central has validated it, and the job drafts the GitHub release.

The same staging and audit run on pull requests as a dry run with a throwaway key: the Ubuntu job of `ci.yml`
(JVM, Android and metadata components) and the `ci:macos` job (every component, including iOS). Locally:

```bash
./scripts/ci/stage-release.sh --dry-run --skip-targets ios,macos   # Linux; on macOS drop --skip-targets
```

## Before the first release

1. **Remove the private runners** (issue #63; before the repository is made public, a fork PR could otherwise edit
   a workflow to run on one). Unregister the Mac, Windows and Linux self-hosted runners, delete `MAC_RUNNER`,
   `WIN_RUNNER` and `LINUX_RUNNER`, and remove the self-hosted routes from the workflows — procedure in
   [`OSS_LAUNCH.md`](OSS_LAUNCH.md) ("Owner steps before the visibility flip", step 1).
   `scripts/ci/verify-hosted-runners.sh` (every PR) fails while the repository is public until that is done.
2. **Central Portal account and namespace.** Sign in at <https://central.sonatype.com>, verify the `com.donfuldauer`
   namespace (DNS TXT record on `donfuldauer.com`), and generate a user token (Account → Generate User Token).
3. **Signing key.** Create a key used only for releases, publish its public half, and export the secret half:

   ```bash
   gpg --quick-generate-key "Kggml releases <you@example.com>" rsa4096 sign 2y
   gpg --keyserver keys.openpgp.org --send-keys <KEY_ID>
   gpg --armor --export-secret-keys <KEY_ID>   # value of SIGNING_KEY
   ```

4. **Repository secrets** (Settings → Secrets and variables → Actions):

   | Secret | Value |
   | --- | --- |
   | `MAVEN_CENTRAL_USERNAME` | Central Portal user token name |
   | `MAVEN_CENTRAL_PASSWORD` | Central Portal user token password |
   | `SIGNING_KEY` | ASCII-armored secret key from step 3 |
   | `SIGNING_KEY_PASSWORD` | its passphrase |
   | `SIGNING_KEY_ID` | optional: the (sub)key id to sign with |

## Dry run

Actions → **Release (Maven Central)** → Run workflow, with **dry run** checked (the default). The run builds,
signs (with `SIGNING_KEY` if set, otherwise a throwaway key) and audits the full release and keeps
`kggml-central-bundle.zip` and the xcframework zips as the `kggml-release-<sha>` workflow artifact. Nothing is
uploaded. To check the bundle with Central's own validation, upload that zip by hand in the portal
(Publish → Publish Component) and drop the deployment afterwards.

To check the full CI matrix on GitHub-hosted runners, run Actions → **CI** → Run workflow on `main` with **runner**
set to `hosted` (the default, `auto`, uses the owner's self-hosted runners that are on); a dispatch runs the Ubuntu,
macOS and Windows jobs.

## Releasing a version

1. Set `VERSION_NAME` in `gradle.properties` (no `-SNAPSHOT`) and merge that change.
2. Tag the merge commit and push the tag:

   ```bash
   git tag -s v0.1.0 -m "Kggml 0.1.0"
   git push origin v0.1.0
   ```

   The workflow fails if the tag and `VERSION_NAME` differ.
3. The run uploads the bundle as a **user-managed** deployment. When it reports `VALIDATED`, review it under
   <https://central.sonatype.com/publishing/deployments> and press **Publish**. Publishing to Maven Central is
   permanent: a version can never be changed or deleted.
4. Publish the draft GitHub release the workflow created (it holds the iOS xcframeworks).

A manual dispatch with **dry run** unchecked on a `v*` tag does the same and can choose `AUTOMATIC` publishing,
which skips step 3.
