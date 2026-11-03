// Scratch verification of the producer's ported platform layer
// (tools/dsv41-prefill/producer-platform.inc). Not part of the build; see
// docs/linux-port-plan.md chunk 3.1c for how it is compiled and run.
#include "ram-output-budget.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace {
using json=nlohmann::json;
using clock_type=std::chrono::steady_clock;
constexpr uint64_t reserve=30ull*1024*1024*1024,receipt_limit=64*1024,chunk=1024*1024;
constexpr uint64_t epoch_ticks=116444736000000000ull;
constexpr uint64_t gib=1024ull*1024*1024;
std::atomic<bool>interrupted{false},memory_abort{false};
void require(bool value,const char*message){if(!value)throw std::runtime_error(message);}
uint64_t add(uint64_t a,uint64_t b){require(b<=UINT64_MAX-a,"overflow");return a+b;}
void stop(){require(!interrupted.load()&&!memory_abort.load(),"cancelled");}
}

#include "producer-platform.inc"

namespace {
std::atomic<int> failures{0};
void ok(bool value,const char*what){
    if(value)std::printf("ok   %s\n",what);
    else{std::printf("FAIL %s\n",what);++failures;}
}
void refused(const char*what,const std::function<void()>&body,bool expect){
    bool caught=false;std::string why;
    try{body();}catch(const std::exception&e){caught=true;why=e.what();}
    if(caught==expect)std::printf("ok   %s%s\n",what,expect?(why.size()>96?(" (refused: "+why.substr(0,96)+"...)").c_str():((" (refused: "+why+")").c_str())):"");
    else{std::printf("FAIL %s (threw=%d expected=%d %s)\n",what,int(caught),int(expect),why.c_str());++failures;}
}
}

int main(int argc,char**argv){
    const std::filesystem::path root=std::filesystem::current_path();
    const std::string secret="dsv41-platform-selftest payload ";
    const std::filesystem::path payload=root/"platform-selftest.bin";
    const std::filesystem::path published=root/"platform-selftest.published";
    ::unlink(payload.c_str());::unlink(published.c_str());

    // 1. volume admission
    ok(native_file::volume(payload)==native_file::volume(root),
        "volume of a path equals the containing directory's device");
    refused("relative path refused",[]{native_file::volume(std::filesystem::path("relative.bin"));},true);
    refused("non-normalized path refused",[]{native_file::volume(std::filesystem::path("/tmp/../tmp/x.bin"));},true);
    refused("RAM-backed volume refused (tmpfs)",[]{native_file::volume(std::filesystem::path("/tmp/x.bin"));},true);

    // 2. exclusive-new output, bounded writes, flush, read-only rehash
    hash incremental;std::string text;
    {
        native_file file;
        file.open(payload,true,4096,0,true);
        refused("a second exclusive-new open of the same path refuses",[&]{native_file other;other.open(payload,true,4096,0,true);},true);
        for(int i=0;i<3;++i){const std::string part=secret+std::to_string(i)+"\n";
            file.write(part.data(),part.size());incremental.update(part.data(),part.size());text+=part;}
        refused("a write past the bounded limit refuses",[&]{
            native_file tight;tight.open(root/"platform-selftest.tight",true,16);
            tight.write("0123456789abcdef0123456789",24);},true);
        ::unlink((root/"platform-selftest.tight").c_str());
        file.flush();
        const auto before=file.current();
        struct stat st{};require(::stat(payload.c_str(),&st)==0,"payload present");
        ok(before.size==uint64_t(text.size())&&before.volume==uint64_t(st.st_dev)&&
            before.id[0]==uint8_t(st.st_ino&0xff)&&before.mtime==uint64_t(st.st_mtim.tv_sec)*10000000ull+
            uint64_t(st.st_mtim.tv_nsec/100)+epoch_ticks,"identity carries st_dev/st_ino/st_mtime as FILETIME ticks");
        ok(describe(before).at("bytes").get<uint64_t>()==uint64_t(text.size()),"describe reports the byte count");
        file.finalize_read_only(incremental.finish());
        ok(true,"closed payload rehashes to the incremental SHA-256");
        ok(file.current().size==uint64_t(text.size()),"final length matches after the read-only re-lease");
        char head[24]{};
        ok(file.read(head,24)==0,"the read-only lease sits at EOF after the certification rehash");
    }

    // 3. read_small agrees with the digest of the file it reads
    {
        const auto read=read_small(payload,text.size()+1);
        ok(read.sha==incremental.finish()&&read.bytes.size()==text.size()&&read.id.size==uint64_t(text.size()),
            "read_small returns the same SHA-256, bytes and identity");
        const auto without=read_small(payload,text.size()+1,false);
        ok(without.bytes.empty()&&!without.sha.empty(),"read_small can hash without retaining bytes");
        refused("read_small refuses a file larger than its admission",[&]{read_small(payload,text.size()-1);},true);
    }

    // 4. no-replace publication
    {
        native_file readonly;readonly.open(payload,false,4096);
        refused("a read-only file cannot publish",[&]{readonly.publish(published);},true);
    }
    {
        native_file receipt;receipt.open(root/"platform-selftest.receipt",true,receipt_limit,0,true);
        const std::string body="{\"schema\":\"selftest\"}\n";
        receipt.write(body.data(),body.size());receipt.flush();
        receipt.publish(published);
        ok(std::filesystem::exists(published)&&!std::filesystem::exists(root/"platform-selftest.receipt"),
            "publication renames the pending receipt in place");
        native_file again;again.open(root/"platform-selftest.receipt2",true,receipt_limit,0,true);
        again.write(body.data(),body.size());again.flush();
        refused("publication over an existing target refuses",[&]{again.publish(published);},true);
        ok(std::filesystem::exists(root/"platform-selftest.receipt2"),"refused publication keeps the pending file");
        refused("publish to a non-sibling path refuses",[&]{again.publish(root/"sub/platform-selftest.published");},true);
    }

    // 5. a symlinked input is refused
    {
        const std::filesystem::path link=root/"platform-selftest.link";
        ::unlink(link.c_str());
        require(::symlink(payload.c_str(),link.c_str())==0,"cannot create test symlink");
        refused("a symlinked input is refused",[&]{native_file f;f.open(link,false,4096);},true);
        ::unlink(link.c_str());
    }

    // 6. held ancestors verify against the named chain
    {
        native_file file;file.open(payload,false,4096);
        file.current();ok(true,"held ancestors verify against the named chain");
        const std::filesystem::path directory=root/"platform-selftest.nested";
        require(std::filesystem::create_directories(directory),"cannot create nested test directory");
        native_file deep;deep.open(directory/"inner.bin",true,64);
        deep.write("x",1);deep.current();ok(true,"a file in a fresh nested chain pins its ancestors");
        ::unlink((directory/"inner.bin").c_str());
        std::filesystem::remove(directory);
    }

    // 7. output byte budget on the real volume
    {
        ok(ds4perf_output_budget::fits(40ull*gib,30ull*gib,0,0,1ull*gib),"output budget admits a 30GiB payload with a 1GiB reserve");
        refused("output budget refuses a payload plus reserve beyond free bytes",[&]{
            native_file f;f.open(payload.parent_path()/"platform-selftest.huge",true,4000ull*gib,0,false,1ull*gib);},true);
        ::unlink((payload.parent_path()/"platform-selftest.huge").c_str());
    }

    // 8. exclusive_writer
    {
        const std::filesystem::path dump=root/"platform-selftest.dump";
        ::unlink(dump.c_str());
        exclusive_writer writer;writer.create(dump);ok(writer.held(),"exclusive writer creates a new file");
        ok(writer.write("abc",3),"exclusive writer writes");
        ok(writer.release(),"exclusive writer closes");
        exclusive_writer second;second.create(dump);ok(!second.held(),"second exclusive create refuses");
        ::unlink(dump.c_str());
    }

    // 9. environment, process identity, executable
    {
        ok(set_environment("SELFTEST_VARIABLE","1")&&std::getenv("SELFTEST_VARIABLE")&&
            !std::strcmp(std::getenv("SELFTEST_VARIABLE"),"1"),"set_environment binds a value the process can read");
        ok(process_id()==uint64_t(::getpid()),"process_id is the pid");
        const auto program=executable();
        ok(program.is_absolute()&&std::filesystem::exists(program)&&program.filename()==std::filesystem::path(argv[0]).filename(),
            "executable() resolves this program");
        ok(file_absent(root/"platform-selftest.absent")&&!file_present(root/"platform-selftest.absent")&&
            file_present(payload),"file_absent/file_present agree with stat");
        ok(remove_file(root/"platform-selftest.receipt2")&&!std::filesystem::exists(root/"platform-selftest.receipt2"),"remove_file unlinks");
        ok(!remove_file(root/"platform-selftest.absent"),"remove_file reports a missing file");
    }

    // 10. RAM, VRAM and cancellation
    {
        const uint64_t available=ram_admission(0);
        ok(available>(1ull<<30),"ram_admission reports the available physical RAM");
        refused("ram_admission refuses a plan beyond available RAM minus 1GiB",[&]{ram_admission(available);},true);
        install_cancellation();
        ::raise(SIGINT);
        ok(interrupted.load(),"SIGINT latches the cancellation flag");
        interrupted.store(false);
        setenv("LLAMADECODE_ABORT_FILE",(root/"platform-selftest.abort").c_str(),1);
        ::unlink((root/"platform-selftest.abort").c_str());
        ok(!abort_file_present(),"no abort file: no graceful abort");
        {std::ofstream flag(root/"platform-selftest.abort");flag<<"x";}
        std::this_thread::sleep_for(std::chrono::milliseconds(260));   // the guard samples at 4 Hz
        ok(abort_file_present(),"a created abort file reports a graceful stop");
        ::unlink((root/"platform-selftest.abort").c_str());
        uint64_t free_bytes=0,total=0;const bool nvml=nvml_free_bytes(free_bytes,total);
        uint64_t budget=0,usage=0;
        local_video_memory(budget,usage);
        std::printf("info nvml_free_bytes=%d device_total=%llu device_used=%llu (CUDA_VISIBLE_DEVICES=%s)\n",
            int(nvml),(unsigned long long)budget,(unsigned long long)usage,
            std::getenv("CUDA_VISIBLE_DEVICES")?"set":"unset");
        ok(true,"video memory queries return without disturbing the device");
        ok(argument("plain-arg")=="plain-arg"&&path_text(payload)==payload.string(),
            "argument/path_text pass bytes through unchanged");
        ok(!decode_library_path().empty()&&decode_library_path().filename()=="ds4-decode.so","decode library resolves beside the executable");
    }

    // 11. the decode library the live session loads (chunk 3.2's loader path)
    {
        const auto wanted=decode_library_path();
        if(std::filesystem::exists(wanted)){
            const library module=open_decode_library();
            ok(module!=nullptr,"open_decode_library maps the decode library beside the executable");
            const char*required[]{"ds4x_open","ds4x_model_map","ds4x_register_weight","ds4x_bind_state",
                "ds4x_set_expert_hook","ds4x_decode","ds4x_close","ds4x_set_affinity"};
            int missing=0;
            for(const char*name:required)if(!library_symbol(module,name)){std::printf("     missing %s\n",name);++missing;}
            ok(missing==0,"every required ds4x_* entry point binds through dlsym");
            const char*optional[]{"ds4x_host_timing_v1","ds4x_host_cache_v1","ds4x_host_cache_v2","ds4x_host_cpu_v1",
                "ds4x_bind_state_ex","ds4x_replace_weight","ds4x_drop_weights","ds4x_set_expert_hook_l2",
                "ds4x_prefill_workspace","ds4x_prefill"};
            std::printf("info optional exports:");
            for(const char*name:optional)std::printf(" %s=%d",name,int(library_symbol(module,name)!=nullptr));
            std::printf("\n");
        } else std::printf("info %s absent beside the selftest: loader check skipped\n",path_text(wanted).c_str());
    }

    ::unlink(payload.c_str());::unlink(published.c_str());
    std::filesystem::remove(root/"sub");
    std::printf("%s: %d failure(s)\n",failures?"SELFTEST FAILED":"SELFTEST PASSED",failures.load());
    return failures?1:0;
}
