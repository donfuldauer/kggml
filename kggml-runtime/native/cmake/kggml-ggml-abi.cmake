# The ggml ABI settings every ggml build in this repository must share: the runtime (kggml-runtime/native) and the
# GPU add-on modules (kggml-runtime-vulkan, kggml-runtime-opencl), which run against the runtime's libggml-base.
#
# GGML_MAX_NAME sizes `char name[GGML_MAX_NAME]` inside `struct ggml_tensor`, ahead of `void * extra`. A backend
# module built with ggml's default (64) would read and write `extra` (ggml-opencl keeps its per-tensor buffers there)
# inside the runtime's 128-byte name instead, and break on tensor names longer than 64 characters.
set(KGGML_GGML_MAX_NAME 128)
