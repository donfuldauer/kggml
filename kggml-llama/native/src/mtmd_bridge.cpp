// mtmd_bridge.cpp - extern "C" libmtmd wrapper: multimodal projector lifecycle and in-memory image evaluation.
// Images are wrapped (raw RGB) or decoded from memory (PNG / JPEG / ...); nothing is ever written to disk.
#include "kggml_llama.h"

#include "guard.hpp"
#include "internal.hpp"

#include "mtmd-helper.h"
#include "mtmd.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

struct kggml_llama_mtmd {
    kggml::llama::LiveCounter live;
    const kggml_llama_model * owner = nullptr; // identity check only
    mtmd_context * ctx = nullptr;

    kggml_llama_mtmd() = default;
    kggml_llama_mtmd(const kggml_llama_mtmd &) = delete;
    kggml_llama_mtmd & operator=(const kggml_llama_mtmd &) = delete;
    ~kggml_llama_mtmd() {
        if (ctx != nullptr) {
            mtmd_free(ctx);
        }
    }
};

struct kggml_llama_mtmd_chunks {
    struct Deleter {
        void operator()(mtmd_input_chunks * c) const noexcept { mtmd_input_chunks_free(c); }
    };
    kggml::llama::LiveCounter live;
    const kggml_llama_mtmd * owner = nullptr; // identity check only
    std::unique_ptr<mtmd_input_chunks, Deleter> chunks;
    int32_t n_images = 0;
};

namespace kggml::llama {
namespace {

kggml_llama_mtmd_params default_mtmd_params() {
    kggml_llama_mtmd_params p{};
    p.backend = KGGML_BACKEND_CPU;
    p.n_threads = 0;
    p.flash_attn = -1;
    p.image_min_tokens = 0;
    p.image_max_tokens = 0;
    p.warmup = 0;
    return p;
}

struct BitmapDeleter {
    void operator()(mtmd_bitmap * b) const noexcept { mtmd_bitmap_free(b); }
};
using BitmapPtr = std::unique_ptr<mtmd_bitmap, BitmapDeleter>;

kggml_status check_mtmd(const kggml_llama_mtmd * mtmd) noexcept {
    if (mtmd == nullptr || mtmd->ctx == nullptr) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, "mtmd is NULL");
    }
    return KGGML_STATUS_OK;
}

/// Wraps / decodes one caller image into an mtmd_bitmap (in memory only).
kggml_status make_bitmap(mtmd_context * ctx, const kggml_llama_image & image, int32_t index, BitmapPtr & out) {
    const std::string where = "image " + std::to_string(index) + ": ";
    if (image.data == nullptr || image.size == 0) {
        return fail(KGGML_STATUS_INVALID_ARGUMENT, (where + "data is NULL or empty").c_str());
    }
    switch (image.format) {
        case KGGML_LLAMA_IMAGE_RGB: {
            if (image.width == 0 || image.height == 0) {
                return fail(KGGML_STATUS_INVALID_ARGUMENT, (where + "width and height must be > 0").c_str());
            }
            const uint64_t expected = static_cast<uint64_t>(image.width) * image.height * 3u;
            if (expected != image.size) {
                return fail(KGGML_STATUS_INVALID_ARGUMENT,
                            (where + "RGB size " + std::to_string(image.size) + " != width * height * 3 = " +
                             std::to_string(expected))
                                .c_str());
            }
            out.reset(mtmd_bitmap_init(image.width, image.height, image.data));
            break;
        }
        case KGGML_LLAMA_IMAGE_ENCODED:
            out.reset(mtmd_helper_bitmap_init_from_buf(ctx, image.data, image.size));
            if (!out) {
                return fail(KGGML_STATUS_INVALID_ARGUMENT,
                            (where + "could not decode the encoded image / audio bytes").c_str());
            }
            break;
        default:
            return fail(KGGML_STATUS_INVALID_ARGUMENT,
                        (where + "unknown format " + std::to_string(image.format)).c_str());
    }
    if (!out) return fail(KGGML_STATUS_OUT_OF_MEMORY, (where + "bitmap allocation failed").c_str());
    return KGGML_STATUS_OK;
}

} // namespace
} // namespace kggml::llama

using kggml::llama::fail;
using kggml::llama::guarded;

extern "C" {

KGGML_LLAMA_API kggml_status kggml_llama_mtmd_default_params(kggml_llama_mtmd_params * params) {
    return guarded(__func__, [&] {
        if (params == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "params is NULL");
        *params = kggml::llama::default_mtmd_params();
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API kggml_status kggml_llama_mtmd_init(kggml_llama_model * model, const char * mmproj_path,
                                                  const kggml_llama_mtmd_params * params,
                                                  kggml_llama_mtmd ** out_mtmd) {
    return guarded(__func__, [&] {
        if (out_mtmd == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_mtmd is NULL");
        *out_mtmd = nullptr;
        if (model == nullptr || model->model == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "model is NULL");
        if (mmproj_path == nullptr || mmproj_path[0] == '\0') {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "mmproj_path is NULL or empty");
        }
        kggml_llama_mtmd_params p = kggml::llama::default_mtmd_params();
        if (params != nullptr) {
            p = *params;
        } else {
            p.backend = model->backend;
        }
        if (p.backend < KGGML_BACKEND_CPU || p.backend > KGGML_BACKEND_COREML) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, ("unknown backend kind " + std::to_string(p.backend)).c_str());
        }
        if (p.backend == KGGML_BACKEND_COREML) {
            return fail(KGGML_STATUS_BACKEND_UNAVAILABLE, "libmtmd has no CoreML backend (use Metal or CPU)");
        }
        if (p.backend != KGGML_BACKEND_CPU) {
            int32_t available = 0;
            if (kggml_runtime_backend_available(p.backend, &available) != KGGML_STATUS_OK || available != 1) {
                return fail(KGGML_STATUS_BACKEND_UNAVAILABLE,
                            "requested GPU backend is not available in this kggml runtime / on this device");
            }
        }
        if (p.n_threads < 0 || p.image_min_tokens < 0 || p.image_max_tokens < 0) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "n_threads / image_min_tokens / image_max_tokens must be >= 0");
        }
        if (p.flash_attn < -1 || p.flash_attn > 1) return fail(KGGML_STATUS_INVALID_ARGUMENT, "flash_attn must be -1, 0 or 1");
        {
            std::FILE * f = std::fopen(mmproj_path, "rb");
            if (f == nullptr) {
                return fail(KGGML_STATUS_MODEL_LOAD_FAILED,
                            (std::string("cannot open mmproj file '") + mmproj_path + "'").c_str());
            }
            std::fclose(f);
        }

        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = p.backend != KGGML_BACKEND_CPU; // decided before any clip backend exists
        mparams.print_timings = false;
        mparams.n_threads = kggml::llama::resolve_threads(p.n_threads);
        mparams.flash_attn_type = static_cast<llama_flash_attn_type>(p.flash_attn);
        mparams.warmup = p.warmup != 0;
        mparams.image_min_tokens = p.image_min_tokens;
        mparams.image_max_tokens = p.image_max_tokens;

        auto mtmd = std::make_unique<kggml_llama_mtmd>();
        mtmd->owner = model;
        mtmd->ctx = mtmd_init_from_file(mmproj_path, model->model, mparams);
        if (mtmd->ctx == nullptr) {
            return fail(KGGML_STATUS_MODEL_LOAD_FAILED,
                        (std::string("libmtmd could not load projector '") + mmproj_path +
                         "' (invalid file or not compatible with the text model)")
                            .c_str());
        }
        *out_mtmd = mtmd.release();
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API void kggml_llama_mtmd_free(kggml_llama_mtmd * mtmd) {
    try {
        delete mtmd;
    } catch (...) {
    }
}

KGGML_LLAMA_API kggml_status kggml_llama_mtmd_get_caps(kggml_llama_mtmd * mtmd, int32_t * out_vision,
                                                      int32_t * out_audio, int32_t * out_audio_sample_rate) {
    return guarded(__func__, [&] {
        if (const auto s = kggml::llama::check_mtmd(mtmd); s != KGGML_STATUS_OK) return s;
        if (out_vision != nullptr) *out_vision = mtmd_support_vision(mtmd->ctx) ? 1 : 0;
        if (out_audio != nullptr) *out_audio = mtmd_support_audio(mtmd->ctx) ? 1 : 0;
        if (out_audio_sample_rate != nullptr) *out_audio_sample_rate = mtmd_get_audio_sample_rate(mtmd->ctx);
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API const char * kggml_llama_mtmd_default_marker(void) {
    try {
        return mtmd_default_marker();
    } catch (...) {
        return "<__media__>";
    }
}

KGGML_LLAMA_API kggml_status kggml_llama_mtmd_tokenize(kggml_llama_mtmd * mtmd, const char * prompt,
                                                      const kggml_llama_image * images, int32_t n_images,
                                                      int32_t add_special, int32_t parse_special,
                                                      kggml_llama_mtmd_chunks ** out_chunks) {
    return guarded(__func__, [&]() -> kggml_status {
        if (out_chunks == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_chunks is NULL");
        *out_chunks = nullptr;
        if (const auto s = kggml::llama::check_mtmd(mtmd); s != KGGML_STATUS_OK) return s;
        if (prompt == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "prompt is NULL");
        if (n_images < 0 || (n_images > 0 && images == nullptr)) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "images is NULL or n_images < 0");
        }

        // The bitmaps only live for this call: mtmd_tokenize() preprocesses them into the chunks.
        std::vector<kggml::llama::BitmapPtr> owned(static_cast<size_t>(n_images));
        std::vector<const mtmd_bitmap *> bitmaps(static_cast<size_t>(n_images));
        for (int32_t i = 0; i < n_images; ++i) {
            if (const auto s = kggml::llama::make_bitmap(mtmd->ctx, images[i], i, owned[static_cast<size_t>(i)]);
                s != KGGML_STATUS_OK) {
                return s;
            }
            bitmaps[static_cast<size_t>(i)] = owned[static_cast<size_t>(i)].get();
        }

        auto result = std::make_unique<kggml_llama_mtmd_chunks>();
        result->owner = mtmd;
        result->n_images = n_images;
        result->chunks.reset(mtmd_input_chunks_init());
        if (!result->chunks) return fail(KGGML_STATUS_OUT_OF_MEMORY, "mtmd_input_chunks_init failed");
        mtmd_input_text text{prompt, add_special != 0, parse_special != 0};
        const int32_t tok = mtmd_tokenize(mtmd->ctx, result->chunks.get(), &text, bitmaps.data(), bitmaps.size());
        if (tok == 1) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT,
                        ("the number of media markers in the prompt does not match n_images = " + std::to_string(n_images))
                            .c_str());
        }
        if (tok != 0) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, ("image preprocessing failed (mtmd_tokenize = " +
                                                        std::to_string(tok) + ")").c_str());
        }
        *out_chunks = result.release();
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API void kggml_llama_mtmd_chunks_free(kggml_llama_mtmd_chunks * chunks) {
    try {
        delete chunks;
    } catch (...) {
    }
}

KGGML_LLAMA_API kggml_status kggml_llama_mtmd_chunks_info(const kggml_llama_mtmd_chunks * chunks, int32_t * out_n_tokens,
                                                         int32_t * out_n_pos, int32_t * out_n_images) {
    return guarded(__func__, [&]() -> kggml_status {
        if (chunks == nullptr || !chunks->chunks) return fail(KGGML_STATUS_INVALID_ARGUMENT, "chunks is NULL");
        const size_t n_tokens = mtmd_helper_get_n_tokens(chunks->chunks.get());
        const llama_pos n_pos = mtmd_helper_get_n_pos(chunks->chunks.get());
        if (out_n_tokens != nullptr) *out_n_tokens = n_tokens > INT32_MAX ? INT32_MAX : static_cast<int32_t>(n_tokens);
        if (out_n_pos != nullptr) *out_n_pos = n_pos;
        if (out_n_images != nullptr) *out_n_images = chunks->n_images;
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API kggml_status kggml_llama_mtmd_eval_chunks(kggml_llama_mtmd * mtmd, kggml_llama_context * context,
                                                         const kggml_llama_mtmd_chunks * chunks, int32_t n_past,
                                                         int32_t seq_id, int32_t logits_last, int32_t * out_n_past) {
    return guarded(__func__, [&]() -> kggml_status {
        if (const auto s = kggml::llama::check_mtmd(mtmd); s != KGGML_STATUS_OK) return s;
        if (const auto s = kggml::llama::check_context(context); s != KGGML_STATUS_OK) return s;
        if (chunks == nullptr || !chunks->chunks) return fail(KGGML_STATUS_INVALID_ARGUMENT, "chunks is NULL");
        if (chunks->owner != mtmd) return fail(KGGML_STATUS_INVALID_ARGUMENT, "chunks were tokenized by another mtmd");
        if (mtmd->owner != context->owner) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "mtmd and context belong to different models");
        }
        if (n_past < 0) return fail(KGGML_STATUS_INVALID_ARGUMENT, "n_past must be >= 0");
        if (const auto s = kggml::llama::check_seq(context, seq_id); s != KGGML_STATUS_OK) return s;
        if (out_n_past == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_n_past is NULL");
        *out_n_past = n_past;

        const int64_t n_ctx = static_cast<int64_t>(llama_n_ctx(context->ctx));
        const int64_t n_pos = mtmd_helper_get_n_pos(chunks->chunks.get());
        if (static_cast<int64_t>(n_past) + n_pos > n_ctx) {
            return fail(KGGML_STATUS_BUFFER_TOO_SMALL, ("prompt needs " + std::to_string(n_pos) + " positions from " +
                                                        std::to_string(n_past) + "; the context holds " +
                                                        std::to_string(n_ctx))
                                                           .c_str());
        }

        context->logits_valid = false;
        const size_t n_chunks = mtmd_input_chunks_size(chunks->chunks.get());
        const int32_t n_batch = static_cast<int32_t>(llama_n_batch(context->ctx));
        llama_pos pos = n_past;
        for (size_t i = 0; i < n_chunks; ++i) {
            // Checked between chunks too: an image encode (clip) is not interruptible by the llama abort callback.
            if (context->abort.load(std::memory_order_acquire)) {
                *out_n_past = pos;
                return fail(KGGML_STATUS_CANCELLED, "multimodal evaluation cancelled");
            }
            const bool last = i + 1 == n_chunks;
            llama_pos next = pos;
            const int32_t ret = mtmd_helper_eval_chunk_single(mtmd->ctx, context->ctx,
                                                              mtmd_input_chunks_get(chunks->chunks.get(), i), pos,
                                                              seq_id, n_batch, last && logits_last != 0, &next);
            if (ret != 0) {
                *out_n_past = pos;
                if (context->abort.load(std::memory_order_acquire)) {
                    return fail(KGGML_STATUS_CANCELLED, "multimodal evaluation cancelled");
                }
                return fail(KGGML_STATUS_NATIVE_FAILURE,
                            ("mtmd_helper_eval_chunk_single failed on chunk " + std::to_string(i) + " (" +
                             std::to_string(ret) + ")").c_str());
            }
            pos = next;
        }
        context->logits_valid = logits_last != 0 && n_chunks > 0;
        *out_n_past = pos;
        return KGGML_STATUS_OK;
    });
}

KGGML_LLAMA_API kggml_status kggml_llama_mtmd_eval(kggml_llama_mtmd * mtmd, kggml_llama_context * context,
                                                  const char * prompt, const kggml_llama_image * images,
                                                  int32_t n_images, int32_t n_past, int32_t seq_id,
                                                  int32_t add_special, int32_t logits_last, int32_t * out_n_past) {
    return guarded(__func__, [&]() -> kggml_status {
        if (out_n_past == nullptr) return fail(KGGML_STATUS_INVALID_ARGUMENT, "out_n_past is NULL");
        *out_n_past = n_past;
        if (const auto s = kggml::llama::check_mtmd(mtmd); s != KGGML_STATUS_OK) return s;
        if (const auto s = kggml::llama::check_context(context); s != KGGML_STATUS_OK) return s;
        if (mtmd->owner != context->owner) {
            return fail(KGGML_STATUS_INVALID_ARGUMENT, "mtmd and context belong to different models");
        }
        kggml_llama_mtmd_chunks * raw = nullptr;
        if (const auto s = kggml_llama_mtmd_tokenize(mtmd, prompt, images, n_images, add_special, 1, &raw);
            s != KGGML_STATUS_OK) {
            return s;
        }
        const std::unique_ptr<kggml_llama_mtmd_chunks> chunks(raw);
        return kggml_llama_mtmd_eval_chunks(mtmd, context, chunks.get(), n_past, seq_id, logits_last, out_n_past);
    });
}

} // extern "C"
