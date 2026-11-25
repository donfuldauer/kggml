// ggml ABI: GGML_MAX_NAME (size of ggml_tensor::name) must be 128 in the runtime AND in every consumer.
//
// stable-diffusion.cpp static_asserts GGML_MAX_NAME >= 128, and the value changes struct ggml_tensor's layout,
// so :kggml-runtime builds ggml with 128 and exports it as a PUBLIC compile definition. This test is compiled
// as a consumer (it only links kggml::runtime / kggml::ggml), so it proves both the propagation (compile-time
// value) and the agreement with the shared libggml-base (a long name survives a ggml_set_name /
// ggml_get_name round trip through the library, and the tensor layout matches what the library writes).
#include "kggml_runtime.h"

#include "ggml.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <string>

namespace {

static_assert(GGML_MAX_NAME == 128, "kggml-runtime must export GGML_MAX_NAME=128 to its consumers");

struct GgmlContext {
    ggml_context * ctx = nullptr;
    GgmlContext() {
        ggml_init_params params{};
        params.mem_size = 1024 * 1024;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ctx = ggml_init(params);
    }
    ~GgmlContext() {
        if (ctx != nullptr) ggml_free(ctx);
    }
};

TEST(AbiTest, GgmlMaxNameIs128) {
    EXPECT_EQ(GGML_MAX_NAME, 128);
    EXPECT_EQ(sizeof(static_cast<ggml_tensor *>(nullptr)->name), static_cast<size_t>(128));
}

TEST(AbiTest, HundredCharTensorNameRoundTripsThroughSharedGgml) {
    GgmlContext g;
    ASSERT_NE(g.ctx, nullptr);
    ggml_tensor * t = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 4);
    ASSERT_NE(t, nullptr);

    std::string name;
    for (int i = 0; i < 100; ++i) name.push_back(static_cast<char>('a' + i % 26));
    ASSERT_EQ(name.size(), 100u);

    ggml_set_name(t, name.c_str()); // strncpy into t->name inside libggml-base (compiled with its own value)
    EXPECT_EQ(std::string(ggml_get_name(t)), name);
    EXPECT_EQ(std::string(t->name), name); // same offset / size as seen from this translation unit
}

TEST(AbiTest, LongestNameIsTruncatedAtMaxNameMinusOne) {
    GgmlContext g;
    ASSERT_NE(g.ctx, nullptr);
    ggml_tensor * t = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 1);
    ASSERT_NE(t, nullptr);

    const std::string too_long(200, 'x');
    ggml_set_name(t, too_long.c_str());
    EXPECT_EQ(std::strlen(ggml_get_name(t)), static_cast<size_t>(GGML_MAX_NAME - 1));
}

TEST(AbiTest, TensorLayoutAgreesWithLibrary) {
    // Fields after `name` (extra) would be misplaced if the library and this TU disagreed on GGML_MAX_NAME.
    GgmlContext g;
    ASSERT_NE(g.ctx, nullptr);
    ggml_tensor * a = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 3, 5);
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(a->extra, nullptr);
    EXPECT_EQ(a->ne[0], 3);
    EXPECT_EQ(a->ne[1], 5);
    EXPECT_EQ(ggml_nelements(a), 15);
    ggml_tensor * b = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 1);
    ASSERT_NE(b, nullptr);
    // Consecutive tensor objects in a no_alloc context are exactly ggml_tensor_overhead() apart, i.e.
    // sizeof(struct ggml_object) + the LIBRARY's sizeof(struct ggml_tensor).
    const auto gap = reinterpret_cast<const char *>(b) - reinterpret_cast<const char *>(a);
    EXPECT_EQ(static_cast<size_t>(gap), ggml_tensor_overhead());
    // struct ggml_object (private to ggml.c) is {size_t, size_t, ptr, enum, char[4]}: 32 bytes on 64-bit
    // targets. A GGML_MAX_NAME mismatch (64 vs 128) would shift this by 64 bytes.
    if (sizeof(void *) == 8) {
        EXPECT_EQ(ggml_tensor_overhead(), sizeof(ggml_tensor) + 32u);
    }
}

} // namespace
