// Context creation: argument validation, backend resolution and model-load failure mapping through the real
// stable-diffusion.cpp loader (missing / empty / garbage / truncated / well-formed-but-not-a-model files). Every
// failure is MODEL_LOAD_FAILED (or INVALID_ARGUMENT / BACKEND_UNAVAILABLE) with a message, never a crash, and
// leaves no runtime reference or context behind.
#include "fixtures.hpp"

#include "gguf.h"

#include <string>
#include <thread>
#include <vector>

using namespace kggml::test;

namespace {

bool contains(const std::string & haystack, const std::string & needle) {
    return haystack.find(needle) != std::string::npos;
}

/// Creates a context from `p` and expects `expected`; checks the out pointer, runtime ref count and live objects.
std::string expect_create_fails(const kggml_diffusion_context_params & p, kggml_status expected) {
    const int32_t refs = kggml_runtime_ref_count();
    const int64_t live = kggml_diffusion_test_live_objects();
    kggml_diffusion_context * c = reinterpret_cast<kggml_diffusion_context *>(0x1);
    const kggml_status s = kggml_diffusion_context_create(&p, &c);
    EXPECT_EQ(s, expected) << kggml_status_name(s) << ": " << kggml_diffusion_last_error();
    EXPECT_EQ(c, nullptr);
    EXPECT_EQ(kggml_runtime_ref_count(), refs) << "runtime reference leaked";
    EXPECT_EQ(kggml_diffusion_test_live_objects(), live) << "context leaked";
    const std::string message = kggml_diffusion_last_error();
    EXPECT_FALSE(message.empty());
    if (c != nullptr && c != reinterpret_cast<kggml_diffusion_context *>(0x1)) kggml_diffusion_context_free(c);
    return message;
}

} // namespace

TEST(DiffusionContext, RequiresAModelPath) {
    kggml_diffusion_context_params p{};
    ASSERT_EQ(kggml_diffusion_context_default_params(&p), KGGML_STATUS_OK);
    EXPECT_TRUE(contains(expect_create_fails(p, KGGML_STATUS_INVALID_ARGUMENT), "model_path or diffusion_model_path"));
    p.model_path = "";
    p.diffusion_model_path = "";
    expect_create_fails(p, KGGML_STATUS_INVALID_ARGUMENT);
}

TEST(DiffusionContext, RejectsOutOfRangeParams) {
    const std::string path = write_gguf("kggml-diffusion-params-", "t", 4);
    kggml_diffusion_context_params p = cpu_params(path);
    p.backend = 42;
    EXPECT_TRUE(contains(expect_create_fails(p, KGGML_STATUS_INVALID_ARGUMENT), "unknown backend kind 42"));
    p = cpu_params(path);
    p.backend = -1;
    expect_create_fails(p, KGGML_STATUS_INVALID_ARGUMENT);
    p = cpu_params(path);
    p.gpu_device = -1;
    EXPECT_TRUE(contains(expect_create_fails(p, KGGML_STATUS_INVALID_ARGUMENT), "gpu_device"));
    p = cpu_params(path);
    p.weight_type = 1000;
    EXPECT_TRUE(contains(expect_create_fails(p, KGGML_STATUS_INVALID_ARGUMENT), "weight_type"));
    p = cpu_params(path);
    p.weight_type = -2;
    expect_create_fails(p, KGGML_STATUS_INVALID_ARGUMENT);
}

TEST(DiffusionContext, MissingFileIsModelLoadFailedNamingThePath) {
    const std::string missing = ::testing::TempDir() + "kggml-diffusion-does-not-exist.gguf";
    kggml_diffusion_context_params p = cpu_params(missing);
    const std::string msg = expect_create_fails(p, KGGML_STATUS_MODEL_LOAD_FAILED);
    EXPECT_TRUE(contains(msg, "cannot open model file")) << msg;
    EXPECT_TRUE(contains(msg, missing)) << msg;

    // Optional component files are checked too.
    const std::string ok = write_gguf("kggml-diffusion-component-", "t", 4);
    p = cpu_params(ok);
    p.vae_path = missing.c_str();
    EXPECT_TRUE(contains(expect_create_fails(p, KGGML_STATUS_MODEL_LOAD_FAILED), "cannot open VAE file"));
    p = cpu_params("");
    p.model_path = nullptr;
    p.diffusion_model_path = missing.c_str();
    EXPECT_TRUE(contains(expect_create_fails(p, KGGML_STATUS_MODEL_LOAD_FAILED), "cannot open diffusion model file"));
}

TEST(DiffusionContext, EmptyFileIsModelLoadFailed) {
    const std::string empty = write_file("kggml-diffusion-empty-", ".gguf", "");
    const std::string msg = expect_create_fails(cpu_params(empty), KGGML_STATUS_MODEL_LOAD_FAILED);
    EXPECT_TRUE(contains(msg, "is empty")) << msg;
}

TEST(DiffusionContext, GarbageBytesAreModelLoadFailed) {
    // Runs stable-diffusion.cpp's real format sniffing (gguf / safetensors / torch zip / torch pickle) on junk.
    for (const char * ext : {".gguf", ".safetensors", ".ckpt", ".bin"}) {
        const std::string garbage =
            write_file("kggml-diffusion-garbage-", ext,
                       "this is not a diffusion model, just some bytes with no recognisable magic number at all");
        const std::string msg = expect_create_fails(cpu_params(garbage), KGGML_STATUS_MODEL_LOAD_FAILED);
        EXPECT_TRUE(contains(msg, "could not load a diffusion model")) << msg;
        EXPECT_TRUE(contains(msg, garbage)) << msg;
    }
}

TEST(DiffusionContext, SafetensorsWithAbsurdHeaderIsModelLoadFailed) {
    // 8-byte little-endian header length larger than the file, then junk: must not allocate it or crash.
    std::string bytes(8, '\0');
    bytes[0] = '\x10';
    bytes[1] = '\x00';
    bytes[2] = '\x00';
    bytes[3] = '\x40';
    bytes += "{\"not\": \"json";
    const std::string path = write_file("kggml-diffusion-badst-", ".safetensors", bytes);
    const std::string msg = expect_create_fails(cpu_params(path), KGGML_STATUS_MODEL_LOAD_FAILED);
    EXPECT_TRUE(contains(msg, "init model loader from file failed")) << msg;
}

TEST(DiffusionContext, TruncatedGgufIsModelLoadFailed) {
    const std::string truncated = write_truncated_gguf();
    const std::string msg = expect_create_fails(cpu_params(truncated), KGGML_STATUS_MODEL_LOAD_FAILED);
    EXPECT_TRUE(contains(msg, truncated)) << msg;
}

TEST(DiffusionContext, WellFormedGgufThatIsNotADiffusionModelIsModelLoadFailed) {
    // Parses fine (written by the shared runtime's gguf_* API) but holds no UNet / DiT / VAE tensors.
    const std::string path = write_gguf("kggml-diffusion-notamodel-", "kggml.test.weight", 64);
    const std::string msg = expect_create_fails(cpu_params(path), KGGML_STATUS_MODEL_LOAD_FAILED);
    EXPECT_TRUE(contains(msg, "could not load a diffusion model")) << msg;
    // stable-diffusion.cpp's own (first) ERROR log line is appended to the message: the file parsed, the model type
    // could not be detected.
    EXPECT_TRUE(contains(msg, "(stable-diffusion.cpp: ")) << msg;
    EXPECT_TRUE(contains(msg, "get sd version from file failed")) << msg;
}

TEST(DiffusionContext, HundredCharTensorNamesParseThroughTheSharedGgml) {
    // ggml's gguf reader (libggml-base of :kggml-runtime) rejects names >= GGML_MAX_NAME. A 100-char tensor name
    // therefore only gets past the gguf parse into stable-diffusion.cpp's model detection when the runtime was
    // built with GGML_MAX_NAME=128 -- the ABI this module requires. The model is still not a diffusion model.
    const std::string name =
        "model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_out.0.kggml_long_name_check";
    ASSERT_GT(name.size(), 64u);
    ASSERT_LT(name.size(), 128u);
    const std::string path = write_gguf("kggml-diffusion-longname-", name, 16);

    // The shared libggml-base reads the full name back.
    gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx = nullptr;
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    ASSERT_NE(g, nullptr);
    ASSERT_EQ(gguf_get_n_tensors(g), 1);
    EXPECT_EQ(std::string(gguf_get_tensor_name(g, 0)), name);
    gguf_free(g);

    // stable-diffusion.cpp's loader accepts the file (the first upstream error is the model-type detection, not
    // "init model loader from file failed" or a name-length error).
    const std::string msg = expect_create_fails(cpu_params(path), KGGML_STATUS_MODEL_LOAD_FAILED);
    EXPECT_TRUE(contains(msg, "get sd version from file failed")) << msg;
    EXPECT_FALSE(contains(msg, "init model loader")) << msg;
    EXPECT_FALSE(contains(msg, "too long")) << msg;
}

TEST(DiffusionContext, CoreMlIsUnavailable) {
    const std::string path = write_gguf("kggml-diffusion-coreml-", "t", 4);
    kggml_diffusion_context_params p = cpu_params(path);
    p.backend = KGGML_BACKEND_COREML;
    EXPECT_TRUE(contains(expect_create_fails(p, KGGML_STATUS_BACKEND_UNAVAILABLE), "CoreML"));
}

TEST(DiffusionContext, UnavailableGpuBackendIsBackendUnavailable) {
    const std::string path = write_gguf("kggml-diffusion-gpu-", "t", 4);
    for (int32_t kind : {KGGML_BACKEND_METAL, KGGML_BACKEND_VULKAN}) {
        ASSERT_EQ(kggml_runtime_init(), KGGML_STATUS_OK);
        int32_t available = 0;
        ASSERT_EQ(kggml_runtime_backend_available(kind, &available), KGGML_STATUS_OK);
        kggml_runtime_shutdown();
        kggml_diffusion_context_params p = cpu_params(path);
        p.backend = kind;
        if (available == 0) {
            expect_create_fails(p, KGGML_STATUS_BACKEND_UNAVAILABLE);
        } else {
            // Device exists: an out-of-range device index is still BACKEND_UNAVAILABLE.
            p.gpu_device = 99;
            EXPECT_TRUE(contains(expect_create_fails(p, KGGML_STATUS_BACKEND_UNAVAILABLE), "device 99 not found"));
        }
    }
}

TEST(DiffusionContext, CpuBackendResolvesToTheGgmlCpuDeviceOnly) {
    kggml_diffusion_context_params p{};
    ASSERT_EQ(kggml_diffusion_context_default_params(&p), KGGML_STATUS_OK);
    char spec[64] = {};
    int32_t uses_gpu = -1;
    const int32_t refs = kggml_runtime_ref_count();
    ASSERT_EQ(kggml_diffusion_test_resolve_backend(&p, spec, sizeof spec, &uses_gpu), KGGML_STATUS_OK)
        << kggml_diffusion_last_error();
    EXPECT_STREQ(spec, "CPU");
    EXPECT_EQ(uses_gpu, 0);
    EXPECT_EQ(kggml_runtime_ref_count(), refs);
    char tiny[2] = {};
    EXPECT_EQ(kggml_diffusion_test_resolve_backend(&p, tiny, sizeof tiny, nullptr), KGGML_STATUS_BUFFER_TOO_SMALL);
}

TEST(DiffusionContext, ConcurrentFailingCreatesAreIndependent) {
    const std::string garbage = write_file("kggml-diffusion-cgarbage-", ".gguf", "garbage garbage garbage garbage");
    const std::string notamodel = write_gguf("kggml-diffusion-cnotamodel-", "x", 8);
    const std::string missing = ::testing::TempDir() + "kggml-diffusion-missing-concurrent.gguf";
    const std::vector<std::string> paths = {garbage, notamodel, missing, garbage, notamodel, missing};
    std::vector<kggml_status> statuses(paths.size(), KGGML_STATUS_OK);
    std::vector<std::string> messages(paths.size());
    std::vector<std::thread> threads;
    for (size_t i = 0; i < paths.size(); ++i) {
        threads.emplace_back([&, i] {
            kggml_diffusion_context_params p{};
            kggml_diffusion_context_default_params(&p);
            p.model_path = paths[i].c_str();
            p.n_threads = 1;
            kggml_diffusion_context * c = nullptr;
            statuses[i] = kggml_diffusion_context_create(&p, &c);
            messages[i] = kggml_diffusion_last_error();
            kggml_diffusion_context_free(c);
        });
    }
    for (auto & t : threads) t.join();
    for (size_t i = 0; i < paths.size(); ++i) {
        EXPECT_EQ(statuses[i], KGGML_STATUS_MODEL_LOAD_FAILED) << paths[i];
        EXPECT_TRUE(contains(messages[i], paths[i])) << "thread " << i << " got another thread's message: " << messages[i];
    }
    EXPECT_EQ(kggml_diffusion_test_live_objects(), 0);
}

TEST(DiffusionContext, StubInfoAndAbortFlag) {
    ContextPtr stub = make_stub();
    kggml_diffusion_context_info info{};
    ASSERT_EQ(kggml_diffusion_context_get_info(stub.get(), &info), KGGML_STATUS_OK);
    EXPECT_EQ(info.backend, KGGML_BACKEND_CPU);
    EXPECT_EQ(info.uses_gpu, 0);
    EXPECT_EQ(info.supports_image_generation, 0);
    EXPECT_EQ(kggml_diffusion_context_get_info(stub.get(), nullptr), KGGML_STATUS_INVALID_ARGUMENT);

    EXPECT_EQ(kggml_diffusion_abort_requested(stub.get()), 0);
    kggml_diffusion_request_abort(stub.get());
    EXPECT_EQ(kggml_diffusion_abort_requested(stub.get()), 1);
    kggml_diffusion_request_abort(stub.get()); // idempotent
    EXPECT_EQ(kggml_diffusion_abort_requested(stub.get()), 1);
    kggml_diffusion_clear_abort(stub.get());
    EXPECT_EQ(kggml_diffusion_abort_requested(stub.get()), 0);

    EXPECT_EQ(kggml_diffusion_test_live_objects(), 1);
    stub.reset();
    EXPECT_EQ(kggml_diffusion_test_live_objects(), 0);
}
