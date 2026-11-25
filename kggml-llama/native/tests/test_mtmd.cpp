// libmtmd entry points: projector load error handling and argument validation (no projector weights are shipped;
// end-to-end vision inference is covered by the :kggml-llama VLM integration tests).
#include "fixtures.hpp"

#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace kggml::test;

TEST(LlamaMtmd, DefaultParams) {
    kggml_llama_mtmd_params p{};
    ASSERT_EQ(kggml_llama_mtmd_default_params(&p), KGGML_STATUS_OK);
    EXPECT_EQ(p.backend, KGGML_BACKEND_CPU);
    EXPECT_EQ(p.n_threads, 0);
    EXPECT_EQ(p.flash_attn, -1);
    EXPECT_EQ(p.warmup, 0);
}

TEST(LlamaMtmd, ProjectorLoadErrors) {
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    kggml_llama_mtmd * mt = reinterpret_cast<kggml_llama_mtmd *>(0x1);

    EXPECT_EQ(kggml_llama_mtmd_init(model.get(), nullptr, nullptr, &mt), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(mt, nullptr);
    EXPECT_EQ(kggml_llama_mtmd_init(nullptr, tiny_model_path().c_str(), nullptr, &mt), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_mtmd_init(model.get(), tiny_model_path().c_str(), nullptr, nullptr),
              KGGML_STATUS_INVALID_ARGUMENT);

    EXPECT_EQ(kggml_llama_mtmd_init(model.get(), (::testing::TempDir() + "kggml-no-mmproj.gguf").c_str(), nullptr, &mt),
              KGGML_STATUS_MODEL_LOAD_FAILED);
    EXPECT_NE(std::string(kggml_llama_last_error()).find("cannot open mmproj"), std::string::npos);
    EXPECT_EQ(kggml_llama_mtmd_init(model.get(), garbage_model_path().c_str(), nullptr, &mt), KGGML_STATUS_MODEL_LOAD_FAILED);
    // A valid GGUF that is a text model, not a projector: libmtmd's clip loader throws, the bridge reports it.
    EXPECT_EQ(kggml_llama_mtmd_init(model.get(), tiny_model_path().c_str(), nullptr, &mt), KGGML_STATUS_MODEL_LOAD_FAILED);
    EXPECT_NE(std::string(kggml_llama_last_error()).find("libmtmd could not load projector"), std::string::npos);
    EXPECT_EQ(mt, nullptr);
}

TEST(LlamaMtmd, ProjectorParamsAreValidated) {
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    kggml_llama_mtmd * mt = nullptr;
    kggml_llama_mtmd_params p{};
    kggml_llama_mtmd_default_params(&p);
    p.backend = KGGML_BACKEND_COREML;
    EXPECT_EQ(kggml_llama_mtmd_init(model.get(), tiny_model_path().c_str(), &p, &mt), KGGML_STATUS_BACKEND_UNAVAILABLE);
    p.backend = 17;
    EXPECT_EQ(kggml_llama_mtmd_init(model.get(), tiny_model_path().c_str(), &p, &mt), KGGML_STATUS_INVALID_ARGUMENT);
    p.backend = KGGML_BACKEND_CPU;
    p.n_threads = -2;
    EXPECT_EQ(kggml_llama_mtmd_init(model.get(), tiny_model_path().c_str(), &p, &mt), KGGML_STATUS_INVALID_ARGUMENT);
    p.n_threads = 0;
    p.flash_attn = 3;
    EXPECT_EQ(kggml_llama_mtmd_init(model.get(), tiny_model_path().c_str(), &p, &mt), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(mt, nullptr);
}

// ---------------------------------------------------------------------------------------------------------------
// End-to-end in-memory image evaluation with the tiny synthetic projector (tiny_model.hpp).
// ---------------------------------------------------------------------------------------------------------------

namespace {

struct MtmdDeleter {
    void operator()(kggml_llama_mtmd * m) const noexcept { kggml_llama_mtmd_free(m); }
};
struct ChunksDeleter {
    void operator()(kggml_llama_mtmd_chunks * c) const noexcept { kggml_llama_mtmd_chunks_free(c); }
};
using MtmdPtr = std::unique_ptr<kggml_llama_mtmd, MtmdDeleter>;
using ChunksPtr = std::unique_ptr<kggml_llama_mtmd_chunks, ChunksDeleter>;

MtmdPtr load_tiny_mmproj(kggml_llama_model * model) {
    kggml_llama_mtmd_params p{};
    kggml_llama_mtmd_default_params(&p);
    p.n_threads = 2;
    kggml_llama_mtmd * mt = nullptr;
    const kggml_status s = kggml_llama_mtmd_init(model, tiny_mmproj_path().c_str(), &p, &mt);
    EXPECT_EQ(s, KGGML_STATUS_OK) << kggml_llama_last_error();
    return MtmdPtr(mt);
}

/// Deterministic packed RGB888 test pattern.
std::vector<uint8_t> rgb_pattern(uint32_t w, uint32_t h, uint32_t seed) {
    std::vector<uint8_t> px(static_cast<size_t>(w) * h * 3);
    for (size_t i = 0; i < px.size(); ++i) px[i] = static_cast<uint8_t>((i * 37u + seed * 101u + (i / 3) * 7u) & 0xFF);
    return px;
}

/// Encodes packed RGB888 as an uncompressed 24-bit BMP (bottom-up, BGR, rows padded to 4 bytes), in memory.
std::vector<uint8_t> bmp_from_rgb(const std::vector<uint8_t> & rgb, uint32_t w, uint32_t h) {
    const uint32_t row = (w * 3 + 3) & ~3u;
    const uint32_t data_size = row * h;
    std::vector<uint8_t> out(54 + data_size, 0);
    auto put32 = [&](size_t at, uint32_t v) {
        for (int i = 0; i < 4; ++i) out[at + i] = static_cast<uint8_t>(v >> (8 * i));
    };
    auto put16 = [&](size_t at, uint16_t v) {
        out[at] = static_cast<uint8_t>(v);
        out[at + 1] = static_cast<uint8_t>(v >> 8);
    };
    out[0] = 'B';
    out[1] = 'M';
    put32(2, static_cast<uint32_t>(out.size()));
    put32(10, 54);
    put32(14, 40);
    put32(18, w);
    put32(22, h);
    put16(26, 1);
    put16(28, 24);
    put32(34, data_size);
    for (uint32_t y = 0; y < h; ++y) {
        const size_t dst = 54 + static_cast<size_t>(h - 1 - y) * row;
        for (uint32_t x = 0; x < w; ++x) {
            const size_t src = (static_cast<size_t>(y) * w + x) * 3;
            out[dst + x * 3 + 0] = rgb[src + 2];
            out[dst + x * 3 + 1] = rgb[src + 1];
            out[dst + x * 3 + 2] = rgb[src + 0];
        }
    }
    return out;
}

kggml_llama_image rgb_image(const std::vector<uint8_t> & px, uint32_t w, uint32_t h) {
    return kggml_llama_image{px.data(), px.size(), KGGML_LLAMA_IMAGE_RGB, w, h};
}

kggml_llama_image encoded_image(const std::vector<uint8_t> & bytes) {
    return kggml_llama_image{bytes.data(), bytes.size(), KGGML_LLAMA_IMAGE_ENCODED, 0, 0};
}

/// Regular files directly in the temp directories (a VLM prompt must not create any).
std::set<std::string> temp_files() {
    std::set<std::string> out;
    for (const auto & dir : {std::filesystem::path(::testing::TempDir()), std::filesystem::temp_directory_path()}) {
        std::error_code ec;
        for (const auto & e : std::filesystem::directory_iterator(dir, ec)) out.insert(e.path().string());
    }
    return out;
}

std::string prompt_with(int n_images) {
    std::string p = "ab";
    for (int i = 0; i < n_images; ++i) p += std::string(" ") + kggml_llama_mtmd_default_marker() + " z";
    return p;
}

/// Evaluates `prompt` + `images` on a fresh context and returns the final logits (empty on failure).
std::vector<float> eval_logits(kggml_llama_model * model, kggml_llama_mtmd * mt, const std::string & prompt,
                               const std::vector<kggml_llama_image> & images, int32_t * out_n_past = nullptr) {
    auto ctx = make_context(model, small_context_params());
    int32_t n_past = -1;
    const kggml_status s = kggml_llama_mtmd_eval(mt, ctx.get(), prompt.c_str(), images.data(),
                                                 static_cast<int32_t>(images.size()), 0, 0, 1, 1, &n_past);
    EXPECT_EQ(s, KGGML_STATUS_OK) << kggml_llama_last_error();
    if (out_n_past != nullptr) *out_n_past = n_past;
    return s == KGGML_STATUS_OK ? last_logits(ctx.get()) : std::vector<float>{};
}

} // namespace

TEST(LlamaMtmd, TinyProjectorLoadsWithVisionCaps) {
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    auto mt = load_tiny_mmproj(model.get());
    ASSERT_TRUE(mt);
    int32_t vision = -1, audio = -1, rate = 0;
    ASSERT_EQ(kggml_llama_mtmd_get_caps(mt.get(), &vision, &audio, &rate), KGGML_STATUS_OK);
    EXPECT_EQ(vision, 1);
    EXPECT_EQ(audio, 0);
    EXPECT_STREQ(kggml_llama_mtmd_default_marker(), "<__media__>");
}

TEST(LlamaMtmd, TokenizeReportsChunkSizes) {
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    auto mt = load_tiny_mmproj(model.get());
    ASSERT_TRUE(mt);
    const auto px = rgb_pattern(kTinyMmprojImageSize, kTinyMmprojImageSize, 1);
    const std::vector<kggml_llama_image> images{rgb_image(px, kTinyMmprojImageSize, kTinyMmprojImageSize),
                                                rgb_image(px, kTinyMmprojImageSize, kTinyMmprojImageSize)};
    kggml_llama_mtmd_chunks * raw = nullptr;
    ASSERT_EQ(kggml_llama_mtmd_tokenize(mt.get(), prompt_with(2).c_str(), images.data(), 2, 1, 1, &raw), KGGML_STATUS_OK)
        << kggml_llama_last_error();
    ChunksPtr chunks(raw);

    // Text-only reference: the same prompt without markers.
    kggml_llama_token text[32];
    int32_t n_text = 0;
    const std::string text_only = "ab z z";
    ASSERT_EQ(kggml_llama_tokenize(model.get(), text_only.data(), text_only.size(), 1, 0, text, 32, &n_text), KGGML_STATUS_OK);

    int32_t n_tokens = 0, n_pos = 0, n_images = 0;
    ASSERT_EQ(kggml_llama_mtmd_chunks_info(chunks.get(), &n_tokens, &n_pos, &n_images), KGGML_STATUS_OK);
    EXPECT_EQ(n_images, 2);
    EXPECT_EQ(n_pos, n_tokens); // no M-RoPE
    // BOS + the text pieces (split at the markers, so possibly tokenized differently) + 4 embeddings per image.
    EXPECT_GT(n_tokens, 2 * kTinyMmprojImageTokens + 3);
    EXPECT_LE(n_tokens, 2 * kTinyMmprojImageTokens + 2 * n_text);
    EXPECT_EQ(kggml_llama_mtmd_chunks_info(nullptr, &n_tokens, &n_pos, &n_images), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_mtmd_chunks_info(chunks.get(), nullptr, nullptr, nullptr), KGGML_STATUS_OK);
}

TEST(LlamaMtmd, RgbImageIsEvaluatedInMemoryDeterministically) {
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    auto mt = load_tiny_mmproj(model.get());
    ASSERT_TRUE(mt);
    const auto px = rgb_pattern(kTinyMmprojImageSize, kTinyMmprojImageSize, 1);
    const std::vector<kggml_llama_image> one{rgb_image(px, kTinyMmprojImageSize, kTinyMmprojImageSize)};

    int32_t n_past = 0;
    const auto a = eval_logits(model.get(), mt.get(), prompt_with(1), one, &n_past);
    ASSERT_EQ(a.size(), static_cast<size_t>(kTinyVocab));
    EXPECT_GT(n_past, kTinyMmprojImageTokens);
    const auto b = eval_logits(model.get(), mt.get(), prompt_with(1), one);
    EXPECT_EQ(a, b);

    const auto other = rgb_pattern(kTinyMmprojImageSize, kTinyMmprojImageSize, 2);
    const auto c = eval_logits(model.get(), mt.get(), prompt_with(1),
                               {rgb_image(other, kTinyMmprojImageSize, kTinyMmprojImageSize)});
    ASSERT_EQ(c.size(), a.size());
    EXPECT_NE(a, c) << "the image content reaches the logits";
}

TEST(LlamaMtmd, EncodedImageDecodesInMemoryLikeTheRawPixels) {
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    auto mt = load_tiny_mmproj(model.get());
    ASSERT_TRUE(mt);
    const auto before = temp_files();
    for (const auto & [w, h] : {std::pair<uint32_t, uint32_t>{28, 28}, {45, 19}}) {
        const auto px = rgb_pattern(w, h, 3);
        const auto bmp = bmp_from_rgb(px, w, h);
        const auto raw = eval_logits(model.get(), mt.get(), prompt_with(1), {rgb_image(px, w, h)});
        const auto decoded = eval_logits(model.get(), mt.get(), prompt_with(1), {encoded_image(bmp)});
        ASSERT_FALSE(raw.empty());
        EXPECT_EQ(raw, decoded) << w << "x" << h << ": BMP is lossless, so the decoded bitmap equals the raw one";
    }
    EXPECT_EQ(temp_files(), before) << "images are decoded in memory, no temporary files";
}

TEST(LlamaMtmd, InvalidImagesAreRejectedWithErrorCodes) {
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    auto mt = load_tiny_mmproj(model.get());
    ASSERT_TRUE(mt);
    const auto px = rgb_pattern(8, 8, 1);
    const std::vector<uint8_t> garbage{'n', 'o', 't', ' ', 'a', 'n', ' ', 'i', 'm', 'a', 'g', 'e'};
    auto bmp = bmp_from_rgb(px, 8, 8);
    bmp[28] = 13; // corrupt header: 13 bits per pixel

    struct Case {
        kggml_llama_image image;
        const char * expect;
    };
    const Case cases[] = {
        {{nullptr, 12, KGGML_LLAMA_IMAGE_ENCODED, 0, 0}, "data is NULL or empty"},
        {{px.data(), 0, KGGML_LLAMA_IMAGE_RGB, 8, 8}, "data is NULL or empty"},
        {{px.data(), px.size(), KGGML_LLAMA_IMAGE_RGB, 0, 8}, "width and height must be > 0"},
        {{px.data(), px.size(), KGGML_LLAMA_IMAGE_RGB, 8, 7}, "RGB size"},
        {{px.data(), px.size(), 7, 8, 8}, "unknown format"},
        {{garbage.data(), garbage.size(), KGGML_LLAMA_IMAGE_ENCODED, 0, 0}, "could not decode"},
        {{bmp.data(), bmp.size(), KGGML_LLAMA_IMAGE_ENCODED, 0, 0}, "could not decode"},
    };
    for (const auto & c : cases) {
        kggml_llama_mtmd_chunks * chunks = reinterpret_cast<kggml_llama_mtmd_chunks *>(0x1);
        EXPECT_EQ(kggml_llama_mtmd_tokenize(mt.get(), prompt_with(1).c_str(), &c.image, 1, 1, 1, &chunks),
                  KGGML_STATUS_INVALID_ARGUMENT)
            << c.expect << " (case " << (&c - cases) << ")";
        EXPECT_EQ(chunks, nullptr);
        EXPECT_NE(std::string(kggml_llama_last_error()).find(c.expect), std::string::npos) << kggml_llama_last_error();
    }

    // Marker / image count mismatch, both ways.
    const auto ok = rgb_pattern(kTinyMmprojImageSize, kTinyMmprojImageSize, 1);
    const kggml_llama_image image = rgb_image(ok, kTinyMmprojImageSize, kTinyMmprojImageSize);
    kggml_llama_mtmd_chunks * chunks = nullptr;
    EXPECT_EQ(kggml_llama_mtmd_tokenize(mt.get(), prompt_with(2).c_str(), &image, 1, 1, 1, &chunks),
              KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_NE(std::string(kggml_llama_last_error()).find("media markers"), std::string::npos);
    EXPECT_EQ(kggml_llama_mtmd_tokenize(mt.get(), "ab", &image, 1, 1, 1, &chunks), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(chunks, nullptr);

    // Argument checks.
    EXPECT_EQ(kggml_llama_mtmd_tokenize(nullptr, "ab", nullptr, 0, 1, 1, &chunks), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_mtmd_tokenize(mt.get(), nullptr, nullptr, 0, 1, 1, &chunks), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_mtmd_tokenize(mt.get(), "ab", nullptr, 1, 1, 1, &chunks), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_mtmd_tokenize(mt.get(), "ab", nullptr, -1, 1, 1, &chunks), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_mtmd_tokenize(mt.get(), "ab", nullptr, 0, 1, 1, nullptr), KGGML_STATUS_INVALID_ARGUMENT);
    // Text-only prompts are fine.
    ASSERT_EQ(kggml_llama_mtmd_tokenize(mt.get(), "ab", nullptr, 0, 1, 1, &chunks), KGGML_STATUS_OK);
    kggml_llama_mtmd_chunks_free(chunks);
    kggml_llama_mtmd_chunks_free(nullptr);
}

TEST(LlamaMtmd, EvalChecksContextCapacityAndOwnership) {
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    auto mt = load_tiny_mmproj(model.get());
    auto other = load_tiny_mmproj(model.get());
    ASSERT_TRUE(mt && other);
    const auto px = rgb_pattern(kTinyMmprojImageSize, kTinyMmprojImageSize, 1);
    const kggml_llama_image image = rgb_image(px, kTinyMmprojImageSize, kTinyMmprojImageSize);
    kggml_llama_mtmd_chunks * raw = nullptr;
    ASSERT_EQ(kggml_llama_mtmd_tokenize(mt.get(), prompt_with(1).c_str(), &image, 1, 1, 1, &raw), KGGML_STATUS_OK);
    ChunksPtr chunks(raw);
    int32_t n_pos = 0;
    ASSERT_EQ(kggml_llama_mtmd_chunks_info(chunks.get(), nullptr, &n_pos, nullptr), KGGML_STATUS_OK);

    auto ctx = make_context(model.get(), small_context_params());
    ASSERT_TRUE(ctx);
    int32_t n_ctx_u = 0;
    {
        uint32_t n = 0;
        ASSERT_EQ(kggml_llama_context_n_ctx(ctx.get(), &n), KGGML_STATUS_OK);
        n_ctx_u = static_cast<int32_t>(n);
    }
    int32_t n_past = -1;
    EXPECT_EQ(kggml_llama_mtmd_eval_chunks(mt.get(), ctx.get(), chunks.get(), n_ctx_u - n_pos + 1, 0, 1, &n_past),
              KGGML_STATUS_BUFFER_TOO_SMALL);
    EXPECT_EQ(n_past, n_ctx_u - n_pos + 1);
    int32_t pos_max = 0;
    ASSERT_EQ(kggml_llama_memory_seq_pos_max(ctx.get(), 0, &pos_max), KGGML_STATUS_OK);
    EXPECT_EQ(pos_max, -1) << "nothing was evaluated";

    EXPECT_EQ(kggml_llama_mtmd_eval_chunks(other.get(), ctx.get(), chunks.get(), 0, 0, 1, &n_past),
              KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_mtmd_eval_chunks(mt.get(), ctx.get(), nullptr, 0, 0, 1, &n_past), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_mtmd_eval_chunks(mt.get(), ctx.get(), chunks.get(), -1, 0, 1, &n_past), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_mtmd_eval_chunks(mt.get(), ctx.get(), chunks.get(), 0, 3, 1, &n_past), KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_llama_mtmd_eval_chunks(mt.get(), ctx.get(), chunks.get(), 0, 0, 1, nullptr), KGGML_STATUS_INVALID_ARGUMENT);

    // Evaluating appends at n_past; the chunks can be evaluated again (e.g. after clearing the cache).
    ASSERT_EQ(kggml_llama_mtmd_eval_chunks(mt.get(), ctx.get(), chunks.get(), 0, 0, 1, &n_past), KGGML_STATUS_OK)
        << kggml_llama_last_error();
    EXPECT_EQ(n_past, n_pos);
    ASSERT_EQ(kggml_llama_memory_seq_pos_max(ctx.get(), 0, &pos_max), KGGML_STATUS_OK);
    EXPECT_EQ(pos_max, n_pos - 1);
    const auto first = last_logits(ctx.get());
    ASSERT_EQ(kggml_llama_memory_clear(ctx.get()), KGGML_STATUS_OK);
    ASSERT_EQ(kggml_llama_mtmd_eval_chunks(mt.get(), ctx.get(), chunks.get(), 0, 0, 1, &n_past), KGGML_STATUS_OK);
    EXPECT_EQ(last_logits(ctx.get()), first);

    // Generation continues from the image prompt with ordinary decodes.
    const kggml_llama_token next = kTinySpaceZ;
    EXPECT_EQ(kggml_llama_decode(ctx.get(), &next, 1, n_past, 0, 1), KGGML_STATUS_OK) << kggml_llama_last_error();
}

TEST(LlamaMtmd, AbortRequestCancelsEvaluation) {
    auto model = load_tiny_model();
    ASSERT_TRUE(model);
    auto mt = load_tiny_mmproj(model.get());
    ASSERT_TRUE(mt);
    const auto px = rgb_pattern(kTinyMmprojImageSize, kTinyMmprojImageSize, 1);
    const kggml_llama_image image = rgb_image(px, kTinyMmprojImageSize, kTinyMmprojImageSize);
    auto ctx = make_context(model.get(), small_context_params());
    kggml_llama_request_abort(ctx.get());
    int32_t n_past = -1;
    EXPECT_EQ(kggml_llama_mtmd_eval(mt.get(), ctx.get(), prompt_with(1).c_str(), &image, 1, 0, 0, 1, 1, &n_past),
              KGGML_STATUS_CANCELLED);
    EXPECT_EQ(n_past, 0);
    kggml_llama_clear_abort(ctx.get());
    EXPECT_EQ(kggml_llama_mtmd_eval(mt.get(), ctx.get(), prompt_with(1).c_str(), &image, 1, 0, 0, 1, 1, &n_past),
              KGGML_STATUS_OK)
        << kggml_llama_last_error();
}

#if defined(KGGML_LLAMA_TEST_HOOKS)
TEST(LlamaMtmd, ProjectorAndChunksAreFreedDeterministically) {
    const int64_t baseline = kggml_llama_test_live_objects();
    {
        auto model = load_tiny_model();
        ASSERT_TRUE(model);
        auto mt = load_tiny_mmproj(model.get());
        ASSERT_TRUE(mt);
        EXPECT_EQ(kggml_llama_test_live_objects(), baseline + 2);
        const auto px = rgb_pattern(kTinyMmprojImageSize, kTinyMmprojImageSize, 1);
        const kggml_llama_image image = rgb_image(px, kTinyMmprojImageSize, kTinyMmprojImageSize);
        kggml_llama_mtmd_chunks * raw = nullptr;
        ASSERT_EQ(kggml_llama_mtmd_tokenize(mt.get(), prompt_with(1).c_str(), &image, 1, 1, 1, &raw), KGGML_STATUS_OK);
        EXPECT_EQ(kggml_llama_test_live_objects(), baseline + 3);
        kggml_llama_mtmd_chunks_free(raw);
        EXPECT_EQ(kggml_llama_test_live_objects(), baseline + 2);
        // A failed tokenize leaves nothing behind.
        const kggml_llama_image bad{px.data(), 3, KGGML_LLAMA_IMAGE_ENCODED, 0, 0};
        EXPECT_EQ(kggml_llama_mtmd_tokenize(mt.get(), prompt_with(1).c_str(), &bad, 1, 1, 1, &raw), KGGML_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(kggml_llama_test_live_objects(), baseline + 2);
        // So does the one-shot eval, on success and on cancellation.
        auto ctx = make_context(model.get(), small_context_params());
        int32_t n_past = 0;
        EXPECT_EQ(kggml_llama_mtmd_eval(mt.get(), ctx.get(), prompt_with(1).c_str(), &image, 1, 0, 0, 1, 1, &n_past),
                  KGGML_STATUS_OK);
        kggml_llama_request_abort(ctx.get());
        EXPECT_EQ(kggml_llama_mtmd_eval(mt.get(), ctx.get(), prompt_with(1).c_str(), &image, 1, n_past, 0, 1, 1, &n_past),
                  KGGML_STATUS_CANCELLED);
        EXPECT_EQ(kggml_llama_test_live_objects(), baseline + 3); // model, mtmd, context
    }
    EXPECT_EQ(kggml_llama_test_live_objects(), baseline);
}
#endif
