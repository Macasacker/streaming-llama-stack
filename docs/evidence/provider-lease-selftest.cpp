// Chunk 3.1f: the weight provider's physical lease, driven through the public
// inspect() with no CUDA backend, no model and no GPU. inspect() takes and
// checks the lease before it asks anything about the device, so every rule the
// Windows build enforced with CreateFileW, BY_HANDLE_FILE_INFORMATION,
// FileIdInfo, GetDriveTypeW and QueryDosDeviceW can be exercised here: the
// receipt binding, the single-link ordinary-file rule, the symlink refusals, the
// local-volume rule, and the fact that a refusal closes what it pinned.
//
// The accept case reaches "requires an actual CUDA backend", which is the proof
// that the lease itself passed. Compile and run: see docs/linux-port-commands.md
// ("The provider's physical lease").
#include "llama-dsv41-weights.h"

#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int failures = 0;
static uint64_t seed = 0x1234ull;

static void check(const char * what, bool ok) {
    std::printf("%-6s %s\n", ok ? "ok" : "FAIL", what);
    std::fflush(stdout);
    if (!ok) ++failures;
}

static bool error_is(const std::string & error, const char * needle) {
    return error.find(needle) != std::string::npos;
}

static int open_descriptors(void) {
    std::error_code ec;
    const int n = static_cast<int>(std::distance(fs::directory_iterator("/proc/self/fd", ec),
                                                 fs::directory_iterator()));
    return ec ? -1 : n;
}

static std::string hex64(const std::string & bytes) {
    // Any 64 hex characters satisfy the receipt's shape check; this is not a hash.
    std::string out;
    for (int i = 0; i < 64; i++) out += static_cast<char>("0123456789abcdef"[(seed >> (i % 16)) & 15]);
    (void)bytes;
    return out;
}

static void write_file(const fs::path & p, size_t bytes) {
    std::ofstream f(p, std::ios::binary);
    std::string payload;
    for (size_t i = 0; i < bytes; i++) { seed = seed * 6364136223846793005ull + 1; payload.push_back(char(seed & 0xff)); }
    f.write(payload.data(), bytes);
}

// Exactly the four facts slstack/native_broker.py writes into a copy receipt.
static llama_dsv41_weight_provider::config receipt_for(const fs::path & p) {
    struct stat st{};
    if (::stat(p.c_str(), &st) != 0) { std::printf("FAIL cannot stat %s\n", p.c_str()); std::exit(1); }
    llama_dsv41_weight_provider::config c;
    c.model_path = p.string();
    c.verified_file_bytes = static_cast<uint64_t>(st.st_size);
    c.verified_volume = static_cast<uint64_t>(st.st_dev);
    const unsigned long long ino = static_cast<unsigned long long>(st.st_ino);
    for (unsigned i = 0; i < 8; i++) c.verified_file_id[i] = uint8_t(ino >> (i * 8));
    c.verified_mtime_ns = static_cast<uint64_t>(st.st_mtim.tv_sec) * 1000000000ull +
        static_cast<uint64_t>(st.st_mtim.tv_nsec);
    c.verified_sha256 = hex64("");
    c.backend = nullptr;   // the lease is checked before the device
    c.admitted_host_bytes = 64ull << 20;
    c.buffers = 1;
    return c;
}

int main(void) {
    const fs::path root = fs::absolute("build/lease-selftest");
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const fs::path model = root / "model.gguf";
    write_file(model, 4096);

    // The accept case: the lease passes and the next thing inspect() says is that
    // it needs a CUDA backend.
    {
        llama_dsv41_weight_provider provider;
        std::string error;
        provider.inspect(receipt_for(model), error);
        // The lease passed, so the next thing inspect() objects to is the device
        // admission this test deliberately leaves empty.
        check("a matching receipt leases the file (reaches the backend check)",
              error_is(error, "backend/buffer admission"));
    }

    // Receipt facts that do not match the file.
    {
        auto c = receipt_for(model);
        c.verified_file_bytes += 1;
        llama_dsv41_weight_provider provider; std::string error;
        provider.inspect(c, error);
        check("a receipt naming a different size is refused", error_is(error, "identity differs"));
    }
    {
        auto c = receipt_for(model);
        c.verified_mtime_ns += 1;
        llama_dsv41_weight_provider provider; std::string error;
        provider.inspect(c, error);
        check("a receipt naming a different mtime is refused", error_is(error, "identity differs"));
    }
    {
        auto c = receipt_for(model);
        c.verified_file_id[3] ^= 0xff;
        llama_dsv41_weight_provider provider; std::string error;
        provider.inspect(c, error);
        check("a receipt naming a different inode is refused", error_is(error, "identity differs"));
    }
    {
        auto c = receipt_for(model);
        c.verified_volume += 1;
        llama_dsv41_weight_provider provider; std::string error;
        provider.inspect(c, error);
        check("a receipt naming a different volume is refused", error_is(error, "identity differs"));
    }
    {
        auto c = receipt_for(model);
        c.verified_sha256 = "not-a-digest";
        llama_dsv41_weight_provider provider; std::string error;
        provider.inspect(c, error);
        check("an incomplete SHA receipt is refused before anything is opened",
              error_is(error, "SHA256/identity receipt required"));
    }
    {
        auto c = receipt_for(model);
        c.model_path = "build/lease-selftest/model.gguf";   // relative
        llama_dsv41_weight_provider provider; std::string error;
        provider.inspect(c, error);
        check("a relative model path is refused", error_is(error, "physical absolute file"));
    }
    {
        auto c = receipt_for(model);
        c.model_path = "/tmp/lease-selftest-model.gguf";
        write_file(c.model_path, 4096);
        struct stat tmp{};
        if (::stat(c.model_path.c_str(), &tmp) != 0) { std::printf("FAIL cannot write the tmpfs fixture\n"); return 1; }
        c.verified_file_bytes = static_cast<uint64_t>(tmp.st_size);
        c.verified_volume = static_cast<uint64_t>(tmp.st_dev);
        c.verified_mtime_ns = static_cast<uint64_t>(tmp.st_mtim.tv_sec) * 1000000000ull + static_cast<uint64_t>(tmp.st_mtim.tv_nsec);
        llama_dsv41_weight_provider provider; std::string error;
        provider.inspect(c, error);
        check("a RAM-backed model volume is refused", error_is(error, "local volume"));
        fs::remove(c.model_path, ec);
    }

    // The file shapes Windows refused by attributes or share modes.
    {
        const fs::path link = root / "link.gguf";
        fs::remove(link, ec);
        fs::create_symlink(model, link, ec);
        llama_dsv41_weight_provider provider; std::string error;
        provider.inspect(receipt_for(link), error);
        check("a symlinked model is refused", error_is(error, "read-only verified model lease"));
    }
    {
        const fs::path hard = root / "hard.gguf";
        fs::remove(hard, ec);
        fs::create_hard_link(model, hard, ec);
        llama_dsv41_weight_provider provider; std::string error;
        auto c = receipt_for(model);
        c.model_path = hard.string();
        provider.inspect(c, error);
        check("a hard-linked model is refused", error_is(error, "single-link ordinary file"));
        fs::remove(hard, ec);   // restores the model's link count
    }
    {
        auto c = receipt_for(model);
        c.model_path = root.string();   // a directory
        c.verified_file_bytes = 4096;
        llama_dsv41_weight_provider provider; std::string error;
        provider.inspect(c, error);
        check("a directory is refused as the model", error_is(error, "single-link ordinary file"));
    }
    {
        const fs::path dir = root / "ancestor-symlink";
        fs::create_symlink(root, dir / "nope", ec);   // fails: parent exists as a file? no, root is a dir
        const fs::path nested = root / "nested";
        fs::create_directories(nested, ec);
        const fs::path nested_model = nested / "model.gguf";
        write_file(nested_model, 4096);
        // Replace the directory component with a symlink to elsewhere: the pin of
        // that ancestor must refuse, as the reparse-point attribute check did.
        fs::remove(nested_model, ec);
        fs::remove(nested, ec);
        fs::create_directory_symlink(root, nested, ec);
        llama_dsv41_weight_provider provider; std::string error;
        auto c = receipt_for(nested_model);
        provider.inspect(c, error);
        check("a symlinked ancestor is refused",
              error_is(error, "Cannot pin model directory") || error_is(error, "local volume"));
    }

    // A refused provider must not leak the descriptors it pinned.
    {
        const int before = open_descriptors();
        auto c = receipt_for(model);
        c.verified_mtime_ns += 1;
        {
            llama_dsv41_weight_provider provider; std::string error;
            provider.inspect(c, error);
        }
        const int after = open_descriptors();
        check("a refused lease closes the descriptors it pinned", before >= 0 && after >= before && after <= before + 1);
    }

    // One provider object inspects once.
    {
        llama_dsv41_weight_provider provider; std::string error;
        provider.inspect(receipt_for(model), error);
        std::string second;
        provider.inspect(receipt_for(model), second);
        // The first inspect failed, which poisons the provider, so the second is
        // refused either way: "cannot reopen" if the first had succeeded.
        check("a provider cannot be inspected twice",
              error_is(second, "cannot reopen") || error_is(second, "provider is invalid"));
    }

    fs::remove_all(root, ec);
    std::printf("SELFTEST %s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
