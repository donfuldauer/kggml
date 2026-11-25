// extern "C" exception -> structured error code mapping (C++ exceptions never cross the FFI boundary), argument
// validation, the thread-local last-error contract and stable-diffusion.cpp log routing.
#include "fixtures.hpp"

#include "ggml.h"
#include "stable-diffusion.h"

#include <cstring>
#include <string>
#include <thread>

using namespace kggml::test;

namespace {

bool contains(const char * haystack, const char * needle) {
    return haystack != nullptr && std::string(haystack).find(needle) != std::string::npos;
}

} // namespace

TEST(DiffusionVersion, ReportsPinnedUpstream) {
    EXPECT_STREQ(kggml_diffusion_version(), KGGML_TEST_EXPECT_DIFFUSION_VERSION);
    EXPECT_STREQ(kggml_diffusion_upstream_version(), KGGML_TEST_EXPECT_UPSTREAM_VERSION);
    EXPECT_STREQ(kggml_diffusion_upstream_commit(), KGGML_TEST_EXPECT_UPSTREAM_COMMIT);
    EXPECT_EQ(std::string(kggml_diffusion_upstream_commit()).size(), 40u);
    // Upstream's own sd_commit() / sd_version() are baked from the same pin (not from the checkout).
    EXPECT_EQ(std::string(kggml_diffusion_upstream_commit()).rfind(sd_commit(), 0), 0u) << sd_commit();
    EXPECT_STREQ(sd_version(), KGGML_TEST_EXPECT_UPSTREAM_VERSION);
    ASSERT_NE(kggml_diffusion_system_info(), nullptr);
    EXPECT_STRNE(kggml_diffusion_system_info(), "");
}

TEST(DiffusionAbi, SharesTheRuntimeGgmlMaxName) {
    // stable-diffusion.cpp static_asserts >= 128; this TU sees the runtime's exported value.
    EXPECT_EQ(GGML_MAX_NAME, 128);
}

TEST(DiffusionErrors, NullArgumentsAreRejectedWithMessages) {
    EXPECT_EQ(kggml_diffusion_context_default_params(nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_TRUE(contains(kggml_diffusion_last_error(), "params is NULL")) << kggml_diffusion_last_error();
    EXPECT_EQ(kggml_diffusion_generate_default_params(nullptr), KGGML_STATUS_INVALID_ARGUMENT);

    kggml_diffusion_context * c = reinterpret_cast<kggml_diffusion_context *>(0x1);
    EXPECT_EQ(kggml_diffusion_context_create(nullptr, &c), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(c, nullptr) << "out pointer must be cleared on failure";
    kggml_diffusion_context_params p{};
    ASSERT_EQ(kggml_diffusion_context_default_params(&p), KGGML_STATUS_OK);
    EXPECT_EQ(kggml_diffusion_context_create(&p, nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_TRUE(contains(kggml_diffusion_last_error(), "out_context is NULL"));

    kggml_diffusion_context_info info{};
    EXPECT_EQ(kggml_diffusion_context_get_info(nullptr, &info), KGGML_STATUS_INVALID_ARGUMENT);

    kggml_diffusion_generate_params gp{};
    ASSERT_EQ(kggml_diffusion_generate_default_params(&gp), KGGML_STATUS_OK);
    Image img;
    img.img.width = 7; // must be zeroed on failure
    EXPECT_EQ(kggml_diffusion_generate(nullptr, &gp, nullptr, nullptr, nullptr, img.out()),
              KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_TRUE(contains(kggml_diffusion_last_error(), "context is NULL"));
    EXPECT_EQ(img.img.width, 0u);
    EXPECT_EQ(img.img.data, nullptr);

    ContextPtr stub = make_stub();
    EXPECT_EQ(kggml_diffusion_generate(stub.get(), &gp, nullptr, nullptr, nullptr, nullptr),
              KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_TRUE(contains(kggml_diffusion_last_error(), "out_image is NULL"));
    EXPECT_EQ(kggml_diffusion_generate(stub.get(), &gp, nullptr, nullptr, nullptr, img.out()),
              KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_TRUE(contains(kggml_diffusion_last_error(), "no model"));

    // Free / abort functions accept NULL; image_free accepts NULL and zeroed images (idempotent).
    kggml_diffusion_context_free(nullptr);
    kggml_diffusion_request_abort(nullptr);
    kggml_diffusion_clear_abort(nullptr);
    EXPECT_EQ(kggml_diffusion_abort_requested(nullptr), 0);
    kggml_diffusion_image_free(nullptr);
    kggml_diffusion_image zero{};
    kggml_diffusion_image_free(&zero);
    kggml_diffusion_image_free(&zero);
}

TEST(DiffusionErrors, DefaultsMatchTheDocumentedContract) {
    kggml_diffusion_context_params p{};
    ASSERT_EQ(kggml_diffusion_context_default_params(&p), KGGML_STATUS_OK);
    EXPECT_EQ(p.backend, KGGML_BACKEND_CPU);
    EXPECT_EQ(p.gpu_device, 0);
    EXPECT_EQ(p.n_threads, 0);
    EXPECT_EQ(p.use_mmap, 1);
    EXPECT_EQ(p.flash_attn, 0);
    EXPECT_EQ(p.offload_params_to_cpu, 0);
    EXPECT_EQ(p.vae_decode_only, 1);
    EXPECT_EQ(p.weight_type, -1);
    EXPECT_EQ(p.model_path, nullptr);

    kggml_diffusion_generate_params g{};
    ASSERT_EQ(kggml_diffusion_generate_default_params(&g), KGGML_STATUS_OK);
    EXPECT_STREQ(g.prompt, "");
    EXPECT_EQ(g.width, 512);
    EXPECT_EQ(g.height, 512);
    EXPECT_EQ(g.sample_steps, 20);
    EXPECT_FLOAT_EQ(g.cfg_scale, 7.0f);
    EXPECT_EQ(g.sample_method, KGGML_DIFFUSION_DEFAULT);
    EXPECT_EQ(g.scheduler, KGGML_DIFFUSION_DEFAULT);
    EXPECT_EQ(g.seed, -1);
    EXPECT_EQ(g.vae_tiling, 0);
}

TEST(DiffusionErrors, SuccessClearsLastErrorAndErrorsAreThreadLocal) {
    EXPECT_EQ(kggml_diffusion_context_default_params(nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_STRNE(kggml_diffusion_last_error(), "");
    std::string other_thread_error = "unset";
    std::thread([&] { other_thread_error = kggml_diffusion_last_error(); }).join();
    EXPECT_EQ(other_thread_error, "");

    kggml_diffusion_context_params p{};
    EXPECT_EQ(kggml_diffusion_context_default_params(&p), KGGML_STATUS_OK);
    EXPECT_STREQ(kggml_diffusion_last_error(), "");

    EXPECT_EQ(kggml_diffusion_context_default_params(nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    kggml_diffusion_clear_last_error();
    EXPECT_STREQ(kggml_diffusion_last_error(), "");
}

// ---------------------------------------------------------------------------------------------------------
// Exception firewall: an exception thrown inside a guarded entry point becomes a status code.
// ---------------------------------------------------------------------------------------------------------

struct FaultCase {
    int32_t fault;
    kggml_status expected;
    const char * message;
};

class DiffusionFaults : public ::testing::TestWithParam<FaultCase> {};

TEST_P(DiffusionFaults, InjectedAtEntryIsMappedAndNeverEscapes) {
    const FaultCase fc = GetParam();
    kggml_diffusion_context_params p{};
    kggml_diffusion_test_inject_fault(fc.fault);
    kggml_status s = KGGML_STATUS_OK;
    EXPECT_NO_THROW(s = kggml_diffusion_context_default_params(&p));
    EXPECT_EQ(s, fc.expected);
    EXPECT_TRUE(contains(kggml_diffusion_last_error(), fc.message)) << kggml_diffusion_last_error();
    EXPECT_TRUE(contains(kggml_diffusion_last_error(), "kggml_diffusion_context_default_params"));
    // One-shot: the next call is clean.
    EXPECT_EQ(kggml_diffusion_context_default_params(&p), KGGML_STATUS_OK);
}

TEST_P(DiffusionFaults, InjectedIntoContextCreateLeaksNothing) {
    const FaultCase fc = GetParam();
    const int32_t refs = kggml_runtime_ref_count();
    const int64_t live = kggml_diffusion_test_live_objects();
    kggml_diffusion_context_params p{};
    ASSERT_EQ(kggml_diffusion_context_default_params(&p), KGGML_STATUS_OK);
    p.model_path = "does-not-matter.gguf";
    kggml_diffusion_context * c = nullptr;
    kggml_diffusion_test_inject_fault(fc.fault);
    EXPECT_EQ(kggml_diffusion_context_create(&p, &c), fc.expected);
    EXPECT_EQ(c, nullptr);
    EXPECT_EQ(kggml_runtime_ref_count(), refs);
    EXPECT_EQ(kggml_diffusion_test_live_objects(), live);
}

TEST_P(DiffusionFaults, ThrownMidGenerationThroughUpstreamFramesIsMapped) {
    // The fault is thrown from the progress dispatcher while stable-diffusion.cpp's pretty_progress() is on the
    // stack, i.e. it unwinds through upstream C++ frames exactly like a real failure inside the sampler would.
    const FaultCase fc = GetParam();
    ContextPtr stub = make_stub();
    Image img;
    kggml_status s = KGGML_STATUS_OK;
    EXPECT_NO_THROW(s = kggml_diffusion_test_simulate_generate(stub.get(), 8, 16, 16, nullptr, nullptr, nullptr,
                                                                /*fault_at_step=*/3, fc.fault, img.out()));
    EXPECT_EQ(s, fc.expected);
    EXPECT_TRUE(contains(kggml_diffusion_last_error(), fc.message)) << kggml_diffusion_last_error();
    EXPECT_EQ(img.img.data, nullptr);
    EXPECT_EQ(img.img.width, 0u);
    // The context stays usable afterwards.
    EXPECT_EQ(kggml_diffusion_test_simulate_generate(stub.get(), 2, 8, 8, nullptr, nullptr, nullptr, -1, 0, img.out()),
              KGGML_STATUS_OK)
        << kggml_diffusion_last_error();
    EXPECT_NE(img.img.data, nullptr);
}

INSTANTIATE_TEST_SUITE_P(AllFaults, DiffusionFaults,
                         ::testing::Values(
                             FaultCase{KGGML_DIFFUSION_TEST_FAULT_BAD_ALLOC, KGGML_STATUS_OUT_OF_MEMORY, "out of memory"},
                             FaultCase{KGGML_DIFFUSION_TEST_FAULT_RUNTIME_ERROR, KGGML_STATUS_NATIVE_FAILURE,
                                       "injected runtime_error"},
                             FaultCase{KGGML_DIFFUSION_TEST_FAULT_LOGIC_ERROR, KGGML_STATUS_NATIVE_FAILURE,
                                       "injected out_of_range"},
                             FaultCase{KGGML_DIFFUSION_TEST_FAULT_NON_STD, KGGML_STATUS_UNKNOWN,
                                       "unknown non-standard C++ exception"}),
                         [](const ::testing::TestParamInfo<FaultCase> & info) {
                             switch (info.param.fault) {
                                 case KGGML_DIFFUSION_TEST_FAULT_BAD_ALLOC: return std::string("BadAlloc");
                                 case KGGML_DIFFUSION_TEST_FAULT_RUNTIME_ERROR: return std::string("RuntimeError");
                                 case KGGML_DIFFUSION_TEST_FAULT_LOGIC_ERROR: return std::string("LogicError");
                                 default: return std::string("NonStd");
                             }
                         });

TEST(DiffusionErrors, FaultsAreThreadLocal) {
    kggml_diffusion_test_inject_fault(KGGML_DIFFUSION_TEST_FAULT_RUNTIME_ERROR);
    kggml_status other = KGGML_STATUS_UNKNOWN;
    std::thread([&] {
        kggml_diffusion_context_params p{};
        other = kggml_diffusion_context_default_params(&p);
    }).join();
    EXPECT_EQ(other, KGGML_STATUS_OK);
    kggml_diffusion_context_params p{};
    EXPECT_EQ(kggml_diffusion_context_default_params(&p), KGGML_STATUS_NATIVE_FAILURE);
}

// ---------------------------------------------------------------------------------------------------------
// Log routing: stable-diffusion.cpp's global logger is silenced, ERROR lines are captured per thread.
// ---------------------------------------------------------------------------------------------------------

TEST(DiffusionLog, ErrorLinesAreCapturedPerThreadOthersIgnored) {
    kggml_diffusion_clear_last_error();
    kggml_diffusion_test_emit_sd_log(SD_LOG_INFO, "just some info");
    EXPECT_STREQ(kggml_diffusion_test_captured_sd_error(), "");
    kggml_diffusion_test_emit_sd_log(SD_LOG_ERROR, "kggml-test-error-line");
    EXPECT_TRUE(contains(kggml_diffusion_test_captured_sd_error(), "kggml-test-error-line"))
        << kggml_diffusion_test_captured_sd_error();
    // No trailing newline is kept.
    const std::string captured = kggml_diffusion_test_captured_sd_error();
    ASSERT_FALSE(captured.empty());
    EXPECT_NE(captured.back(), '\n');

    std::string other = "unset";
    std::thread([&] { other = kggml_diffusion_test_captured_sd_error(); }).join();
    EXPECT_EQ(other, "");

    // Any guarded call starts with a clean slate.
    kggml_diffusion_context_params p{};
    ASSERT_EQ(kggml_diffusion_context_default_params(&p), KGGML_STATUS_OK);
    EXPECT_STREQ(kggml_diffusion_test_captured_sd_error(), "");
}

TEST(DiffusionErrors, GenerateParamsAreValidatedBeforeAnyWork) {
    ContextPtr stub = make_stub();
    kggml_diffusion_generate_params base{};
    ASSERT_EQ(kggml_diffusion_generate_default_params(&base), KGGML_STATUS_OK);
    struct Case {
        const char * what;
        void (*mutate)(kggml_diffusion_generate_params &);
        const char * message;
    };
    const Case cases[] = {
        {"null prompt", [](kggml_diffusion_generate_params & p) { p.prompt = nullptr; }, "prompt is NULL"},
        {"zero width", [](kggml_diffusion_generate_params & p) { p.width = 0; }, "multiples of 8"},
        {"odd height", [](kggml_diffusion_generate_params & p) { p.height = 513; }, "multiples of 8"},
        {"negative width", [](kggml_diffusion_generate_params & p) { p.width = -512; }, "multiples of 8"},
        {"huge", [](kggml_diffusion_generate_params & p) { p.width = 32768; }, "<= 16384"},
        {"zero steps", [](kggml_diffusion_generate_params & p) { p.sample_steps = 0; }, "sample_steps"},
        {"bad sampler", [](kggml_diffusion_generate_params & p) { p.sample_method = SAMPLE_METHOD_COUNT; }, "sample_method"},
        {"bad sampler low", [](kggml_diffusion_generate_params & p) { p.sample_method = -2; }, "sample_method"},
        {"bad scheduler", [](kggml_diffusion_generate_params & p) { p.scheduler = SCHEDULER_COUNT; }, "scheduler"},
    };
    for (const Case & c : cases) {
        kggml_diffusion_generate_params p = base;
        c.mutate(p);
        Image img;
        EXPECT_EQ(kggml_diffusion_generate(stub.get(), &p, nullptr, nullptr, nullptr, img.out()),
                  KGGML_STATUS_INVALID_ARGUMENT)
            << c.what;
        EXPECT_TRUE(contains(kggml_diffusion_last_error(), c.message)) << c.what << ": " << kggml_diffusion_last_error();
    }
    // Valid params reach the model check (the stub has none).
    Image img;
    EXPECT_EQ(kggml_diffusion_generate(stub.get(), &base, nullptr, nullptr, nullptr, img.out()),
              KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_TRUE(contains(kggml_diffusion_last_error(), "no model"));
}
