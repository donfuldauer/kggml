// tiny_model.hpp - writes a structurally valid, randomly initialized whisper ggml model for tests.
//
// The model uses real whisper.cpp hyper-parameter layout (English vocab size, 80 mel bins, 1500 audio
// positions) but a 16-wide state and one encoder / decoder layer, so it is ~1.7 MB and a 1 s inference
// runs in milliseconds. Its output text is meaningless; tests use it to drive the real whisper.cpp
// encode / decode path (abort callbacks, backend selection, padding) without shipping model weights.
#pragma once

#include <string>

namespace kggml::test {

/// Path of the tiny model, generated once per test process under the GoogleTest temp dir.
const std::string & tiny_model_path();

/// Path of a file that is not a whisper model (bad magic).
const std::string & garbage_model_path();

} // namespace kggml::test
