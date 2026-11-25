// Progress / abort plumbing. kggml_diffusion_test_simulate_generate() runs the real generation machinery of
// kggml_diffusion_generate() with the sampler replaced by a loop over stable-diffusion.cpp's own pretty_progress(),
// so every callback below travels: upstream pretty_progress() -> process-wide sd progress hook -> the wrapper's
// per-thread dispatcher -> the caller's callbacks, and an abort unwinds back through the upstream frame.
#include "fixtures.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace kggml::test;

namespace {

struct Recorder {
    std::vector<int32_t> steps;
    std::vector<int32_t> totals;
    int32_t abort_at = -1;            // abort callback returns 1 once this step has been reported
    int32_t abort_polls = 0;
    std::thread::id thread;
    bool wrong_thread = false;
    kggml_diffusion_context * request_abort_on = nullptr; // request_abort() from another thread at abort_at
};

void record_progress(int32_t step, int32_t total, float /*seconds*/, void * user_data) {
    auto * r = static_cast<Recorder *>(user_data);
    if (std::this_thread::get_id() != r->thread) r->wrong_thread = true;
    r->steps.push_back(step);
    r->totals.push_back(total);
    if (r->request_abort_on != nullptr && step == r->abort_at) {
        std::thread([c = r->request_abort_on] { kggml_diffusion_request_abort(c); }).join();
    }
}

int32_t abort_when_reached(void * user_data) {
    auto * r = static_cast<Recorder *>(user_data);
    ++r->abort_polls;
    return r->abort_at >= 0 && r->request_abort_on == nullptr && !r->steps.empty() && r->steps.back() >= r->abort_at
               ? 1
               : 0;
}

kggml_status simulate(kggml_diffusion_context * c, int32_t steps, Recorder & r, Image & img, int32_t w = 16,
                      int32_t h = 8) {
    r.thread = std::this_thread::get_id();
    return kggml_diffusion_test_simulate_generate(c, steps, w, h, record_progress, abort_when_reached, &r, -1, 0,
                                                  img.out());
}

std::vector<int32_t> iota(int32_t from, int32_t to) {
    std::vector<int32_t> v;
    for (int32_t i = from; i <= to; ++i) v.push_back(i);
    return v;
}

} // namespace

TEST(DiffusionProgress, ReportsEveryStepInOrderWithTotalsAndUserData) {
    ContextPtr c = make_stub();
    Recorder r;
    Image img;
    ASSERT_EQ(simulate(c.get(), 12, r, img), KGGML_STATUS_OK) << kggml_diffusion_last_error();
    EXPECT_EQ(r.steps, iota(0, 12)); // 0 = sampling started, then 1..total after each step
    EXPECT_EQ(r.totals, std::vector<int32_t>(13, 12));
    EXPECT_FALSE(r.wrong_thread) << "callbacks must run on the generating thread";
    // The abort callback is polled once before starting and once per progress point.
    EXPECT_EQ(r.abort_polls, 1 + 13);
    EXPECT_STREQ(kggml_diffusion_last_error(), "");
}

TEST(DiffusionProgress, ImageIsHandedOverAsRgb8) {
    ContextPtr c = make_stub();
    Recorder r;
    Image img;
    ASSERT_EQ(simulate(c.get(), 1, r, img, 40, 24), KGGML_STATUS_OK) << kggml_diffusion_last_error();
    ASSERT_NE(img.img.data, nullptr);
    EXPECT_EQ(img.img.width, 40u);
    EXPECT_EQ(img.img.height, 24u);
    EXPECT_EQ(img.img.channels, 3u);
    for (uint32_t y = 0; y < img.img.height; ++y) {
        for (uint32_t x = 0; x < img.img.width; ++x) {
            const uint8_t * p = img.img.data + (y * img.img.width + x) * 3;
            ASSERT_EQ(p[0], x & 255);
            ASSERT_EQ(p[1], y & 255);
            ASSERT_EQ(p[2], (x + y) & 255);
        }
    }
    kggml_diffusion_image_free(img.out());
    EXPECT_EQ(img.img.data, nullptr);
    EXPECT_EQ(img.img.width, 0u);
}

TEST(DiffusionProgress, NullCallbacksAreFine) {
    ContextPtr c = make_stub();
    Image img;
    EXPECT_EQ(kggml_diffusion_test_simulate_generate(c.get(), 5, 8, 8, nullptr, nullptr, nullptr, -1, 0, img.out()),
              KGGML_STATUS_OK);
    EXPECT_NE(img.img.data, nullptr);
}

class DiffusionAbortAtStep : public ::testing::TestWithParam<int32_t> {};

TEST_P(DiffusionAbortAtStep, AbortCallbackStopsAtThatStep) {
    const int32_t k = GetParam();
    ContextPtr c = make_stub();
    Recorder r;
    r.abort_at = k;
    Image img;
    ASSERT_EQ(simulate(c.get(), 10, r, img), KGGML_STATUS_CANCELLED);
    EXPECT_EQ(r.steps, iota(0, k)) << "no progress may be reported after the abort";
    EXPECT_EQ(img.img.data, nullptr);
    EXPECT_EQ(img.img.width, 0u);
    const std::string msg = kggml_diffusion_last_error();
    EXPECT_NE(msg.find("aborted at progress step " + std::to_string(k) + "/10"), std::string::npos) << msg;
    // Not latched: a per-call abort callback does not poison the context.
    EXPECT_EQ(kggml_diffusion_abort_requested(c.get()), 0);
    Recorder again;
    Image img2;
    EXPECT_EQ(simulate(c.get(), 3, again, img2), KGGML_STATUS_OK) << kggml_diffusion_last_error();
    EXPECT_EQ(again.steps, iota(0, 3));
}

INSTANTIATE_TEST_SUITE_P(Steps, DiffusionAbortAtStep, ::testing::Values(0, 1, 5, 9, 10));

TEST(DiffusionAbort, AbortCallbackBeforeStartCancelsWithoutProgress) {
    ContextPtr c = make_stub();
    Image img;
    std::vector<int32_t> seen;
    auto progress = [](int32_t step, int32_t, float, void * ud) { static_cast<std::vector<int32_t> *>(ud)->push_back(step); };
    auto always = [](void *) -> int32_t { return 1; };
    EXPECT_EQ(kggml_diffusion_test_simulate_generate(c.get(), 4, 8, 8, progress, always, &seen, -1, 0, img.out()),
              KGGML_STATUS_CANCELLED);
    EXPECT_TRUE(seen.empty());
    EXPECT_EQ(img.img.data, nullptr);
}

TEST(DiffusionAbort, LatchedRequestCancelsUntilCleared) {
    ContextPtr c = make_stub();
    kggml_diffusion_request_abort(c.get());
    Recorder r;
    Image img;
    EXPECT_EQ(simulate(c.get(), 4, r, img), KGGML_STATUS_CANCELLED);
    EXPECT_TRUE(r.steps.empty());
    EXPECT_NE(std::string(kggml_diffusion_last_error()).find("kggml_diffusion_clear_abort"), std::string::npos);
    EXPECT_EQ(simulate(c.get(), 4, r, img), KGGML_STATUS_CANCELLED) << "the request stays latched";

    kggml_diffusion_clear_abort(c.get());
    Recorder r2;
    EXPECT_EQ(simulate(c.get(), 4, r2, img), KGGML_STATUS_OK);
    EXPECT_EQ(r2.steps, iota(0, 4));
}

TEST(DiffusionAbort, RequestFromAnotherThreadMidGenerationStopsAtTheNextStepBoundary) {
    ContextPtr c = make_stub();
    Recorder r;
    r.abort_at = 6;
    r.request_abort_on = c.get(); // another thread latches the abort while step 6 is being reported
    Image img;
    EXPECT_EQ(simulate(c.get(), 20, r, img), KGGML_STATUS_CANCELLED);
    EXPECT_EQ(r.steps, iota(0, 6));
    EXPECT_EQ(kggml_diffusion_abort_requested(c.get()), 1);
    EXPECT_EQ(img.img.data, nullptr);
}

TEST(DiffusionAbort, ThrowingCallbacksAbortInsteadOfEscaping) {
    ContextPtr c = make_stub();
    Image img;
    auto throwing_progress = [](int32_t step, int32_t, float, void *) {
        if (step == 2) throw std::runtime_error("callback bug");
    };
    kggml_status s = KGGML_STATUS_OK;
    EXPECT_NO_THROW(s = kggml_diffusion_test_simulate_generate(c.get(), 5, 8, 8, throwing_progress, nullptr, nullptr,
                                                                -1, 0, img.out()));
    EXPECT_EQ(s, KGGML_STATUS_CANCELLED);

    auto throwing_abort = [](void *) -> int32_t { throw 7; };
    EXPECT_NO_THROW(s = kggml_diffusion_test_simulate_generate(c.get(), 5, 8, 8, nullptr, throwing_abort, nullptr, -1,
                                                                0, img.out()));
    EXPECT_EQ(s, KGGML_STATUS_CANCELLED);
    EXPECT_EQ(img.img.data, nullptr);
}

TEST(DiffusionAbort, ReentrantGenerateFromACallbackIsRejected) {
    ContextPtr outer = make_stub();
    ContextPtr inner = make_stub();
    struct State {
        kggml_diffusion_context * inner;
        kggml_status nested = KGGML_STATUS_OK;
        int calls = 0;
    } st{inner.get()};
    auto progress = [](int32_t step, int32_t, float, void * ud) {
        auto * s = static_cast<State *>(ud);
        ++s->calls;
        if (step == 1) {
            kggml_diffusion_image nested{};
            s->nested = kggml_diffusion_test_simulate_generate(s->inner, 2, 8, 8, nullptr, nullptr, nullptr, -1, 0,
                                                               &nested);
            kggml_diffusion_image_free(&nested);
        }
    };
    Image img;
    EXPECT_EQ(kggml_diffusion_test_simulate_generate(outer.get(), 3, 8, 8, progress, nullptr, &st, -1, 0, img.out()),
              KGGML_STATUS_OK);
    EXPECT_EQ(st.nested, KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(st.calls, 4) << "the outer generation keeps its own dispatch after the rejected nested call";
}

TEST(DiffusionAbort, ConcurrentContextsDoNotCrossTalk) {
    // N contexts generate at the same time on N threads, each with its own step count, user_data and abort point.
    // The process-wide upstream progress hook must deliver every step only to the generation of the calling thread.
    constexpr int kThreads = 6;
    std::vector<ContextPtr> contexts;
    for (int i = 0; i < kThreads; ++i) contexts.push_back(make_stub());

    struct PerThread {
        Recorder rec;
        kggml_status status = KGGML_STATUS_UNKNOWN;
        int32_t steps = 0;
    };
    std::vector<PerThread> results(kThreads);
    std::mutex m;
    std::condition_variable cv;
    int ready = 0;

    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i] {
            PerThread & pt = results[i];
            pt.steps = 50 + 13 * i;
            pt.rec.abort_at = (i % 3 == 0) ? 7 + i : -1; // every third thread aborts itself
            {
                std::unique_lock<std::mutex> lock(m);
                ++ready;
                cv.notify_all();
                cv.wait(lock, [&] { return ready == kThreads; });
            }
            Image img;
            pt.status = simulate(contexts[i].get(), pt.steps, pt.rec, img, 8 * (i + 1), 8);
            if (pt.status == KGGML_STATUS_OK) {
                EXPECT_EQ(img.img.width, static_cast<uint32_t>(8 * (i + 1)));
            }
        });
    }
    for (auto & t : threads) t.join();

    for (int i = 0; i < kThreads; ++i) {
        const PerThread & pt = results[i];
        SCOPED_TRACE("thread " + std::to_string(i));
        EXPECT_FALSE(pt.rec.wrong_thread);
        EXPECT_EQ(pt.rec.totals, std::vector<int32_t>(pt.rec.steps.size(), pt.steps)) << "saw another context's total";
        if (pt.rec.abort_at >= 0) {
            EXPECT_EQ(pt.status, KGGML_STATUS_CANCELLED);
            EXPECT_EQ(pt.rec.steps, iota(0, pt.rec.abort_at));
        } else {
            EXPECT_EQ(pt.status, KGGML_STATUS_OK);
            EXPECT_EQ(pt.rec.steps, iota(0, pt.steps));
        }
        EXPECT_EQ(kggml_diffusion_abort_requested(contexts[i].get()), 0);
    }
}

TEST(DiffusionAbort, LatchedAbortOnOneContextDoesNotAffectAnother) {
    ContextPtr a = make_stub();
    ContextPtr b = make_stub();
    kggml_diffusion_request_abort(a.get());
    Recorder ra, rb;
    Image ia, ib;
    EXPECT_EQ(simulate(a.get(), 3, ra, ia), KGGML_STATUS_CANCELLED);
    EXPECT_EQ(simulate(b.get(), 3, rb, ib), KGGML_STATUS_OK);
    EXPECT_EQ(rb.steps, iota(0, 3));
}

TEST(DiffusionAbort, RejectsBadSimulationArguments) {
    ContextPtr c = make_stub();
    Image img;
    EXPECT_EQ(kggml_diffusion_test_simulate_generate(c.get(), 0, 8, 8, nullptr, nullptr, nullptr, -1, 0, img.out()),
              KGGML_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(kggml_diffusion_test_simulate_generate(nullptr, 1, 8, 8, nullptr, nullptr, nullptr, -1, 0, img.out()),
              KGGML_STATUS_INVALID_ARGUMENT);
}
