// Version reporting: kggml runtime version and pinned upstream ggml version / commit.
#include "kggml_runtime.h"

#include "ggml.h"

#include <gtest/gtest.h>

#include <cctype>
#include <string>

namespace {

TEST(VersionsTest, RuntimeVersionMatchesBuild) {
    ASSERT_NE(kggml_runtime_version(), nullptr);
    EXPECT_STRNE(kggml_runtime_version(), "");
    EXPECT_STREQ(kggml_runtime_version(), KGGML_TEST_EXPECT_RUNTIME_VERSION);
}

TEST(VersionsTest, GgmlVersionMatchesPinnedRelease) {
    ASSERT_NE(kggml_runtime_ggml_version(), nullptr);
    EXPECT_STRNE(kggml_runtime_ggml_version(), "");
    EXPECT_STREQ(kggml_runtime_ggml_version(), KGGML_TEST_EXPECT_GGML_VERSION);
    // Same answer as the shared libggml-base the runtime is dynamically linked against.
    EXPECT_STREQ(kggml_runtime_ggml_version(), ggml_version());
}

TEST(VersionsTest, GgmlCommitIsFullPinnedSha) {
    const std::string commit = kggml_runtime_ggml_commit();
    EXPECT_EQ(commit, KGGML_TEST_EXPECT_GGML_COMMIT);
    ASSERT_EQ(commit.size(), 40u);
    for (char c : commit) {
        EXPECT_TRUE(std::isxdigit(static_cast<unsigned char>(c)) && !std::isupper(static_cast<unsigned char>(c)))
            << "unexpected character '" << c << "' in " << commit;
    }
}

TEST(VersionsTest, GgmlBuildCommitIsPrefixOfPinnedShaWhenKnown) {
    std::string build = kggml_runtime_ggml_build_commit();
    ASSERT_FALSE(build.empty());
    EXPECT_EQ(build, std::string(ggml_commit()));
    if (build == "unknown") {
        GTEST_SKIP() << "ggml built without git metadata (e.g. source tarball)";
    }
    const std::string dirty = "-dirty";
    if (build.size() > dirty.size() && build.compare(build.size() - dirty.size(), dirty.size(), dirty) == 0) {
        build.resize(build.size() - dirty.size());
    }
    EXPECT_EQ(std::string(KGGML_TEST_EXPECT_GGML_COMMIT).rfind(build, 0), 0u)
        << "ggml was built from " << build << " but the pin is " << KGGML_TEST_EXPECT_GGML_COMMIT;
}

} // namespace
