// extern "C" exception -> structured error code mapping (C++ exceptions never cross the FFI boundary), argument
// validation and the thread-local last-error contract.
#include "fixtures.hpp"

#include <ostream>
#include <string>
#include <thread>

using namespace kggml::test;

TEST(LlamaVersion, ReportsPinnedUpstream) {
    EXPECT_STREQ(kggml_llama_version(), KGGML_TEST_EXPECT_LLAMA_VERSION);
    EXPECT_STREQ(kggml_llama_upstream_version(), KGGML_TEST_EXPECT_UPSTREAM_VERSION);
    EXPECT_STREQ(kggml_llama_upstream_commit(), KGGML_TEST_EXPECT_UPSTREAM_COMMIT);
    EXPECT_EQ(std::string(kggml_llama_upstream_commit()).size(), 40u);
    ASSERT_NE(kggml_llama_system_info(), nullptr);
    EXPECT_NE(std::string(kggml_llama_system_info()).find("CPU"), std::string::npos);
    EXPECT_STREQ(kggml_llama_mtmd_default_marker(), "<__media__>");
}

TEST(LlamaErrors, NullArgumentsAreRejectedWithMessages) {
    EXPECT_EQ(kggml_llama_model_default_params(nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(std::string(kggml_llama_last_error()).find("params is NULL"), std::string::npos);
    EXPECT_EQ(kggml_llama_context_default_params(nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_sampler_default_params(nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_mtmd_default_params(nullptr), KGGML_STATUS_INVALID_ARGUMENT);

    kggml_llama_model * m = reinterpret_cast<kggml_llama_model *>(0x1);
    EXPECT_EQ(kggml_llama_model_load(nullptr, nullptr, &m), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(m, nullptr) << "out pointer must be cleared on failure";
    EXPECT_EQ(kggml_llama_model_load("", nullptr, &m), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_model_load("x.gguf", nullptr, nullptr), KGGML_STATUS_INVALID_ARGUMENT);

    kggml_llama_context * c = nullptr;
    EXPECT_EQ(kggml_llama_context_create(nullptr, nullptr, &c), KGGML_STATUS_INVALID_ARGUMENT);
    kggml_llama_model_info info{};
    EXPECT_EQ(kggml_llama_model_get_info(nullptr, &info), KGGML_STATUS_INVALID_ARGUMENT);
    const kggml_llama_token t = 1;
    EXPECT_EQ(kggml_llama_decode(nullptr, &t, 1, 0, 0, 1), KGGML_STATUS_INVALID_ARGUMENT);
    const float * logits = nullptr;
    int32_t n = 0;
    EXPECT_EQ(kggml_llama_get_logits(nullptr, &logits, &n), KGGML_STATUS_INVALID_ARGUMENT);
    size_t sz = 0;
    EXPECT_EQ(kggml_llama_state_size(nullptr, -1, &sz), KGGML_STATUS_INVALID_ARGUMENT);
    kggml_llama_token out = 0;
    EXPECT_EQ(kggml_llama_sampler_sample(nullptr, nullptr, -1, &out), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_sampler_accept(nullptr, 0, 0), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_sampler_reset(nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    int32_t np = 0;
    EXPECT_EQ(kggml_llama_mtmd_eval(nullptr, nullptr, "", nullptr, 0, 0, 0, 1, 1, &np), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_mtmd_get_caps(nullptr, nullptr, nullptr, nullptr), KGGML_STATUS_INVALID_ARGUMENT);

    // Free / abort functions accept NULL.
    kggml_llama_model_free(nullptr);
    kggml_llama_context_free(nullptr);
    kggml_llama_sampler_free(nullptr);
    kggml_llama_mtmd_free(nullptr);
    kggml_llama_request_abort(nullptr);
    kggml_llama_clear_abort(nullptr);
    EXPECT_EQ(kggml_llama_abort_requested(nullptr), 0);
    EXPECT_EQ(kggml_llama_token_is_eog(nullptr, 2), 0);
}

TEST(LlamaErrors, SuccessClearsLastErrorAndErrorsAreThreadLocal) {
    EXPECT_EQ(kggml_llama_model_default_params(nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_STRNE(kggml_llama_last_error(), "");
    std::string other_thread_error = "unset";
    std::thread([&] { other_thread_error = kggml_llama_last_error(); }).join();
    EXPECT_EQ(other_thread_error, "");

    kggml_llama_model_params p{};
    EXPECT_EQ(kggml_llama_model_default_params(&p), KGGML_STATUS_OK);
    EXPECT_STREQ(kggml_llama_last_error(), "");
    EXPECT_EQ(p.backend, KGGML_BACKEND_CPU);
    EXPECT_EQ(p.gpu_layers, -1);
    EXPECT_EQ(p.use_mmap, 1);

    EXPECT_EQ(kggml_llama_model_default_params(nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    kggml_llama_clear_last_error();
    EXPECT_STREQ(kggml_llama_last_error(), "");
}

TEST(LlamaErrors, InvalidParametersAreRejectedBeforeLoading) {
    kggml_llama_model_params p{};
    kggml_llama_model_default_params(&p);
    kggml_llama_model * m = nullptr;

    p.backend = 42;
    EXPECT_EQ(kggml_llama_model_load(tiny_model_path().c_str(), &p, &m), KGGML_STATUS_INVALID_ARGUMENT);
    p.backend = KGGML_BACKEND_CPU;
    p.gpu_device = -1;
    EXPECT_EQ(kggml_llama_model_load(tiny_model_path().c_str(), &p, &m), KGGML_STATUS_INVALID_ARGUMENT);
    p.gpu_device = 0;
    p.gpu_layers = -2;
    EXPECT_EQ(kggml_llama_model_load(tiny_model_path().c_str(), &p, &m), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(m, nullptr);
}

#if defined(KGGML_LLAMA_TEST_HOOKS)

namespace {

struct FaultCase {
    int32_t fault;
    kggml_status expected;
    const char * message_part;
};

void PrintTo(const FaultCase & c, std::ostream * os) {
    *os << "fault=" << c.fault << " expected=" << kggml_status_name(c.expected);
}

class FaultMapping : public ::testing::TestWithParam<FaultCase> {};

} // namespace

TEST_P(FaultMapping, ParamsEntryPointMapsInjectedException) {
    const auto c = GetParam();
    kggml_llama_context_params p{};
    kggml_llama_test_inject_fault(c.fault);
    EXPECT_EQ(kggml_llama_context_default_params(&p), c.expected);
    EXPECT_NE(std::string(kggml_llama_last_error()).find(c.message_part), std::string::npos) << kggml_llama_last_error();
    EXPECT_NE(std::string(kggml_llama_last_error()).find("kggml_llama_context_default_params"), std::string::npos);
    // One-shot: the next call succeeds and clears the error.
    EXPECT_EQ(kggml_llama_context_default_params(&p), KGGML_STATUS_OK);
    EXPECT_STREQ(kggml_llama_last_error(), "");
}

TEST_P(FaultMapping, ModelLoadMapsInjectedExceptionAndLeaksNothing) {
    const auto c = GetParam();
    const int64_t live = kggml_llama_test_live_objects();
    const int32_t refs = kggml_runtime_ref_count();
    kggml_llama_model * m = nullptr;
    kggml_llama_test_inject_fault(c.fault);
    EXPECT_EQ(kggml_llama_model_load(tiny_model_path().c_str(), nullptr, &m), c.expected);
    EXPECT_EQ(m, nullptr);
    EXPECT_EQ(kggml_llama_test_live_objects(), live);
    EXPECT_EQ(kggml_runtime_ref_count(), refs);
}

TEST_P(FaultMapping, DecodeAndSamplerEntryPointsMapInjectedException) {
    const auto c = GetParam();
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    auto ctx = make_context(model.get(), small_context_params());
    ASSERT_TRUE(ctx);
    const auto prompt = letter_prompt(4);

    kggml_llama_test_inject_fault(c.fault);
    EXPECT_EQ(kggml_llama_decode(ctx.get(), prompt.data(), 4, 0, 0, 1), c.expected);
    EXPECT_NE(std::string(kggml_llama_last_error()).find("kggml_llama_decode"), std::string::npos);
    // The context stays usable after a trapped exception.
    EXPECT_EQ(kggml_llama_decode(ctx.get(), prompt.data(), 4, 0, 0, 1), KGGML_STATUS_OK) << kggml_llama_last_error();

    kggml_llama_sampler * s = nullptr;
    kggml_llama_test_inject_fault(c.fault);
    EXPECT_EQ(kggml_llama_sampler_create(model.get(), nullptr, &s), c.expected);
    EXPECT_EQ(s, nullptr);

    size_t size = 0;
    kggml_llama_test_inject_fault(c.fault);
    EXPECT_EQ(kggml_llama_state_size(ctx.get(), -1, &size), c.expected);
}

INSTANTIATE_TEST_SUITE_P(
    AllFaults, FaultMapping,
    ::testing::Values(FaultCase{KGGML_LLAMA_TEST_FAULT_BAD_ALLOC, KGGML_STATUS_OUT_OF_MEMORY, "out of memory"},
                      FaultCase{KGGML_LLAMA_TEST_FAULT_RUNTIME_ERROR, KGGML_STATUS_NATIVE_FAILURE, "injected runtime_error"},
                      FaultCase{KGGML_LLAMA_TEST_FAULT_LOGIC_ERROR, KGGML_STATUS_NATIVE_FAILURE, "injected out_of_range"},
                      FaultCase{KGGML_LLAMA_TEST_FAULT_NON_STD, KGGML_STATUS_UNKNOWN, "non-standard"}));

TEST(LlamaErrors, FaultsAreThreadLocal) {
    kggml_llama_test_inject_fault(KGGML_LLAMA_TEST_FAULT_RUNTIME_ERROR);
    kggml_status other = KGGML_STATUS_UNKNOWN;
    std::thread([&] {
        kggml_llama_sampler_params p{};
        other = kggml_llama_sampler_default_params(&p);
    }).join();
    EXPECT_EQ(other, KGGML_STATUS_OK);
    kggml_llama_sampler_params p{};
    EXPECT_EQ(kggml_llama_sampler_default_params(&p), KGGML_STATUS_NATIVE_FAILURE);
}

#endif // KGGML_LLAMA_TEST_HOOKS
