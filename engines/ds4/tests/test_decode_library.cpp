/* Gate for ds4-decode.so: the producer binds this ABI with dlopen.
 *
 * The decode library is a second compilation of the engine (ds4_ext.o, ds4_cuda_ext.o, built
 * with -DDS4_EXTERNAL_DECODE=1). Nothing in this fork loads it today - the producer-side
 * dlopen wiring is chunk 3.2 - so this harness is what proves the library loads and exports
 * the entry points the caller expects.
 *
 * Two entry points are exercised without a GPU and without a model file:
 *   ds4x_host_cpu_v1  installs the caller's ggml CPU dot kernel and starts the thread pool;
 *   ds4x_open         must refuse loudly for anything that is not a DeepSeek V4.1 GGUF.
 *
 * Usage: test_decode_library <path-to-ds4-decode.so> <path-to-a-non-model-file>
 */
#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_external_host_cache.h"

namespace {

const char *EXPORTS[] = {
    "ds4x_host_cpu_v1", "ds4x_vision_file", "ds4x_open", "ds4x_model_map", "ds4x_register_weight",
    "ds4x_replace_weight", "ds4x_drop_weights", "ds4x_bind_state_ex", "ds4x_bind_state",
    "ds4x_set_expert_hook", "ds4x_set_expert_hook_l2", "ds4x_set_affinity", "ds4x_decode",
    "ds4x_prefill_workspace", "ds4x_prefill", "ds4x_close", "ds4x_host_cache_v1",
    "ds4x_host_cache_v2", "ds4x_host_timing_v1",
};

/* The dot hook is only installed, never invoked here: ds4x_open refuses before any decode. */
void stub_dot(int n, float *s, size_t type, const void *x, size_t tx, const void *y, size_t ty, int scale) {
    (void)n; (void)type; (void)x; (void)tx; (void)y; (void)ty; (void)scale;
    *s = 0.0f;
}

int failures = 0;

void check(bool ok, const char *fmt, ...) {
    if (ok) return;
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "FAIL: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    failures++;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <ds4-decode.so> <non-model-file>\n", argv[0]);
        return 2;
    }
    void *handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "FAIL: dlopen(%s): %s\n", argv[1], dlerror());
        return 1;
    }
    for (const char *name : EXPORTS) {
        check(dlsym(handle, name) != nullptr, "export %s is missing", name);
    }
    if (failures) {
        dlclose(handle);
        fprintf(stderr, "DECODE_LIBRARY: %d check(s) FAILED\n", failures);
        return 1;
    }

    auto host_cpu = (int (*)(ds4_external_dot))dlsym(handle, "ds4x_host_cpu_v1");
    check(host_cpu(stub_dot) == 1, "ds4x_host_cpu_v1 refused the caller's dot kernel");

    auto open_fn = (int (*)(const char *, uint32_t, char *, size_t))dlsym(handle, "ds4x_open");
    auto close_fn = (void (*)(void))dlsym(handle, "ds4x_close");
    auto model_map = (const void *(*)(void))dlsym(handle, "ds4x_model_map");

    char error[256] = {0};
    check(open_fn(argv[2], 1, error, sizeof(error)) == 0, "ds4x_open accepted a file that is not a V4.1 GGUF");
    check(error[0] != '\0', "ds4x_open refused without saying why");
    fprintf(stderr, "DECODE_LIBRARY refusal: %s\n", error);

    /* Nothing may be open after a refusal, and closing an unopened library must be safe. */
    check(model_map() == nullptr, "ds4x_model_map returned a map with no model open");
    close_fn();
    check(model_map() == nullptr, "ds4x_model_map still returns a map after close");

    dlclose(handle);
    fprintf(stdout, "DECODE_LIBRARY: PASS %s exports %zu entry points\n", argv[1],
            sizeof(EXPORTS) / sizeof(EXPORTS[0]));
    return 0;
}
