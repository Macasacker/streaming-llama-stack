// Producer platform layer. Included by tools/dsv41-prefill/main.cpp after the
// producer's own policy helpers (require, stop, the bounded constants) and
// before live-session.inc, so both see these names in the same namespace.
//
// This is the port of the Win32 layer the producer ran before this fork went
// Linux-only. Each line names what the Windows call became; the mapping is the
// one docs/linux-port-plan.md Phase 2 lists. The Windows code itself is deleted,
// so this is the only platform layer the producer has:
//
//   CreateFileW, ReadFile, WriteFile, FlushFileBuffers   open, read, write, fsync
//   BY_HANDLE_FILE_INFORMATION + FILE_ID_INFO            fstat: st_dev, st_ino, st_size, st_mtime
//   GetDriveTypeW + QueryDosDeviceW                      fstatfs: refuse network and RAM-backed volumes
//   GetDiskFreeSpaceExW                                  fstatvfs
//   SetFileInformationByHandle(FileRenameInfo, no replace)  renameat2(RENAME_NOREPLACE)
//   BCrypt SHA-256                                       dsv41_boundary::sha256_native (portable state)
//   GlobalMemoryStatusEx                                 llama_dsv41_host_memory::read (MemAvailable)
//   GetModuleFileNameW                                   /proc/self/exe
//   SetConsoleCtrlHandler                                sigaction(SIGINT, SIGTERM)
//   DXGI local video-memory segment budget               CUDA device total/free (no WDDM segments)
//   nvml.dll through LoadLibraryExW                      libnvidia-ml.so.1 through dlopen
//
// Two guarantees the old Win32 layer had have no POSIX twin and are deliberately
// not faked: share modes (Linux cannot deny another process opening the same
// file), and fixed-vs-removable media (no statfs indicator). Identity rechecks
// before and after every file operation stand in for the first; network and
// RAM-backed volumes are refused by filesystem type.
#include <cerrno>
#include <dlfcn.h>
#include <fcntl.h>
#include <linux/fs.h>
#include <linux/magic.h>
#include <signal.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include "boundary-capture.h"          // dsv41_boundary::sha256_native
#include "llama-dsv41-host-memory.h"   // the RAM figure available_ram reports
#include "producer-platform.h"         // argument, path_text, native_path
#include "producer-wire.h"             // the session handle exclusive_writer uses

namespace {
using producer_text::argument;using producer_text::native_path;using producer_text::path_text;

// Incremental SHA-256 of the bytes the producer writes or binds. update() is
// bounded to one chunk; finish() finalizes and the object is done. It is the
// portable state the boundary capture diagnostics already carry - the same
// digest BCrypt's CNG path produced.
struct hash {
    dsv41_boundary::sha256_native digest;
    void update(const void*data,size_t bytes){require(bytes<=chunk,"Native SHA256 update failed");digest.update(data,bytes);}
    std::string finish(){return digest.hex();}
};

// Physical identity of a held file: the volume it lives on, its inode, its
// length, and its modification time in FILETIME ticks - 100 ns since 1601-01-01,
// the unit the receipts and the weight lease compare, so a receipt written here
// has the same shape as one written by the Windows build.
struct identity {uint64_t volume=0,size=0,mtime=0;std::array<uint8_t,16>id{};};
std::string id_hex(const std::array<uint8_t,16>&id){std::ostringstream s;s<<std::hex<<std::setfill('0');for(auto b:id)s<<std::setw(2)<<unsigned(b);return s.str();}
json describe(const identity&id){return {{"volume_serial_u64",id.volume},{"file_id_le_hex",id_hex(id.id)},{"bytes",id.size},{"mtime_filetime_ticks",id.mtime}};}
bool same(const identity&a,const identity&b){return a.volume==b.volume&&a.id==b.id&&a.size==b.size&&a.mtime==b.mtime;}

// st_ino occupies the low 8 bytes of the 16-byte id field the receipts carry,
// little-endian; the high 8 stay zero on this platform.
identity inspect(int fd){
    struct stat st{};
    require(fstat(fd,&st)==0,"Cannot inspect native file identity");
    require(S_ISREG(st.st_mode)&&st.st_nlink==1&&st.st_size>=0,"Producer requires a physical single-link ordinary file");
    identity out;out.volume=uint64_t(st.st_dev);out.size=uint64_t(st.st_size);
    const unsigned long long ino=static_cast<unsigned long long>(st.st_ino);
    for(int i=0;i<8;++i)out.id[size_t(i)]=uint8_t(ino>>(8*i));
    require(st.st_mtim.tv_sec>=0,"File modification time predates the Unix epoch");
    out.mtime=uint64_t(st.st_mtim.tv_sec)*10000000ull+uint64_t(st.st_mtim.tv_nsec/100)+epoch_ticks;
    return out;
}

// No unlink, replacement, truncation or cleanup operation exists here. The sole
// rename publishes our new flushed pending receipt without replacing it. Held
// directory descriptors pin the ancestor chain, and before each file action those
// identities are rebound to the current named chain as well.
class native_file {
    std::filesystem::path path;
    std::vector<int>parents;
    std::vector<std::filesystem::path>names;
    int file=-1;
    identity owned{};
    uint64_t limit=0,pending=0,disk_floor=reserve;
    bool output=false,can_publish=false,flushed=false;

    void parents_ok()const{
        for(size_t i=0;i<parents.size();++i){
            struct stat held{},named{};
            require(fstat(parents[i],&held)==0&&S_ISDIR(held.st_mode),"Held output ancestor changed");
            // O_NOFOLLOW refuses a symlink ancestor, as the reparse-point check did.
            const int probe=::open(names[i].c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
            const bool valid=probe>=0&&fstat(probe,&named)==0&&S_ISDIR(named.st_mode)&&
                held.st_dev==named.st_dev&&held.st_ino==named.st_ino;
            if(probe>=0)close(probe);
            require(valid,"Named output ancestor identity changed");
        }
    }

public:
    native_file()=default;
    ~native_file(){if(file!=closed_marker())close_held();for(auto h:parents)release(h);}
    native_file(const native_file&)=delete;native_file&operator=(const native_file&)=delete;

    // The selected volume's identity, and the rule that it is a local disk.
    static uint64_t volume(const std::filesystem::path&p){
        require(p.is_absolute()&&p==p.lexically_normal()&&p.has_filename(),"Producer paths must be absolute and normalized");
        struct statfs probe{};
        // The directory, not the file: volume() is also called for an output
        // path that does not exist yet.
        require(statfs(p.parent_path().c_str(),&probe)==0,"Cannot inspect producer volume");
        // The old rule was DRIVE_FIXED on a physical volume, not UNC or SUBST.
        // Linux has no fixed/removable indicator, so remote and RAM-backed types
        // are refused by their magic numbers; an unknown local type is accepted.
        const long type=long(probe.f_type);
        const long refused[]={NFS_SUPER_MAGIC,FUSE_SUPER_MAGIC,0x5346544D/*SMB*/,0xfe534d42/*SMB2*/,0xff53ff53/*CIFS*/,
            0x53464146/*AFS*/,0x010209/*9P*/,0x7461636f/*OCFS2*/,0x01161970/*GFS2*/,0x564c/*NCP*/,
            TMPFS_MAGIC,RAMFS_MAGIC,0x516194e9/*devtmpfs*/};
        for(const long magic:refused)require(type!=magic,"Producer requires a fixed local volume, not a network or RAM-backed one");
        struct ::stat root{};
        require(::stat(p.parent_path().c_str(),&root)==0&&S_ISDIR(root.st_mode),"Cannot identify producer volume");
        return uint64_t(root.st_dev);
    }

    void open(const std::filesystem::path&p,bool create,uint64_t expected,uint64_t outstanding=0,bool publishable=false,uint64_t output_floor=reserve){
        require(file==-1&&parents.empty(),"Native producer file is single-use");
        path=p;volume(path);
        require(path_text(path.filename()).find(':')==std::string::npos,"Alternate data streams are forbidden");
        std::vector<std::filesystem::path>chain;for(auto d=path.parent_path();;d=d.parent_path()){chain.push_back(d);if(d==d.root_path())break;}
        std::reverse(chain.begin(),chain.end());
        for(const auto&d:chain){
            const int h=::open(d.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
            require(h>=0,"Cannot lease physical producer ancestor");
            parents.push_back(h);names.push_back(d);parents_ok();
        }
        limit=expected;pending=outstanding;disk_floor=output_floor;output=create;can_publish=publishable;parents_ok();if(create)budget();parents_ok();
        // No share-mode twin: O_EXCL is the exclusive-new admission, and the
        // identity rechecks refuse a file another process replaced.
        const int flags=(create?(O_RDWR|O_CREAT|O_EXCL):(O_RDONLY))|O_NOFOLLOW|O_CLOEXEC;
        file=::open(path.c_str(),flags,0644);
        require(file>=0,"Cannot open exclusive-new output or read-only physical input");
        owned=inspect(file);
        require(create?owned.size==0:owned.size<=limit,"Producer file exceeds its bounded byte limit");check();
    }
    void check()const{parents_ok();require(same(owned,inspect(file)),"Native producer file identity/size/mtime changed");}
    void budget()const{
        parents_ok();
        struct statvfs probe{};
        require(statvfs(path.parent_path().c_str(),&probe)==0,"Cannot inspect actual output-volume free bytes");
        const uint64_t free_bytes=uint64_t(probe.f_bavail)*uint64_t(probe.f_frsize);
        require(ds4perf_output_budget::fits(free_bytes,limit,owned.size,pending,disk_floor),
            "Producer output/carry/other pending bytes exceed actual-volume budget plus selected disk reserve");
    }
    void outstanding(uint64_t bytes){pending=bytes;}
    identity current()const{check();return owned;}
    size_t read(void*dst,size_t bytes){
        require(!output&&bytes<=chunk,"Invalid bounded input read");check();
        const ssize_t got=::read(file,dst,bytes);
        require(got>=0,"Producer input read failed");check();return size_t(got);
    }
    void write(const void*src,size_t bytes){
        require(output&&bytes<=chunk&&owned.size<=limit&&bytes<=limit-owned.size,"Producer output exceeds planned bounded size");
        check();budget();flushed=false;
        const unsigned char*p=static_cast<const unsigned char*>(src);size_t left=bytes;
        while(left){const ssize_t sent=::write(file,p,left);require(sent>0,"Native output write failed; incomplete file retained");p+=size_t(sent);left-=size_t(sent);}
        const auto after=inspect(file);require(after.volume==owned.volume&&after.id==owned.id&&after.size==owned.size+bytes,"Native output ownership or length changed");owned=after;check();
    }
    void flush(){
        check();
        require(fsync(file)==0,"Native output flush failed; output is not certified");
        owned=inspect(file);check();flushed=true;
    }
    void finalize_read_only(const std::string&expected_sha){
        require(output&&flushed,"Payload must be flushed before closing its writer");check();const auto before=owned;
        require(close(file)==0,"Cannot close completed payload writer");file=-1;parents_ok();
        const int readonly=::open(path.c_str(),O_RDONLY|O_NOFOLLOW|O_CLOEXEC);
        require(readonly>=0,"Cannot acquire completed payload readonly lease");file=readonly;
        const auto final=inspect(file);
        require(final.volume==before.volume&&final.id==before.id&&final.size==before.size,"Payload identity changed during writer-close/read-lease gap");
        owned=final;output=false;check();hash digest;std::vector<uint8_t>buffer(chunk);uint64_t count=0;
        for(;;){stop();const auto got=read(buffer.data(),buffer.size());if(!got)break;digest.update(buffer.data(),got);count+=got;}
        require(count==owned.size&&digest.finish()==expected_sha,"Closed payload differs from its incremental SHA; no receipt may publish");check();
    }
    void publish(const std::filesystem::path&target){
        require(output&&can_publish&&flushed,"Only an owned fully flushed pending receipt can publish");check();
        require(target.is_absolute()&&target==target.lexically_normal()&&target.has_filename()&&target.parent_path()==path.parent_path()&&
            path_text(target.filename()).find(':')==std::string::npos,"Receipt publication requires a sibling ordinary path");
        // Prepare every allocation/validation before publication: a concurrent
        // user file at the target name can never be overwritten.
        std::filesystem::path destination=target;
        parents_ok();stop();
        // RENAME_NOREPLACE fails with EEXIST on an occupied target instead of
        // publishing over it, which is what ReplaceIfExists=FALSE means.
        require(renameat2(AT_FDCWD,path.c_str(),AT_FDCWD,destination.c_str(),RENAME_NOREPLACE)==0,
            "No-replace receipt publication failed; pending file retained");
        // No fallible operation follows the atomic publication.
        path.swap(destination);
    }

private:
    static int closed_marker(){return -1;}
    void close_held(){close(file);file=-1;}
    static void release(int h){close(h);}
};

struct contents {std::vector<uint8_t>bytes;std::string sha;identity id;};
contents read_small(const std::filesystem::path&path,uint64_t maximum,bool retain=true){
    native_file file;file.open(path,false,maximum);contents out;out.id=file.current();hash digest;
    std::vector<uint8_t>buffer(chunk);uint64_t read=0;
    for(;;){stop();const auto count=file.read(buffer.data(),buffer.size());if(!count)break;read+=count;require(read<=maximum,"Input grew beyond bounded read admission");
        digest.update(buffer.data(),count);if(retain)out.bytes.insert(out.bytes.end(),buffer.begin(),buffer.begin()+count);}
    require(read==out.id.size,"Input byte count changed while reading");out.sha=digest.finish();return out;
}

std::filesystem::path executable(){
    std::vector<char>name(4096);
    for(;;){
        const ssize_t n=readlink("/proc/self/exe",name.data(),name.size());
        require(n>=0,"Cannot identify actual producer binary");
        if(size_t(n)<name.size())return std::filesystem::path(std::string(name.data(),size_t(n)));
        if(name.size()>=1<<20)require(false,"Cannot identify actual producer binary");
        name.resize(name.size()*2);
    }
}
uint64_t process_id(){return uint64_t(getpid());}
// Command arguments arrive as UTF-16 here and as bytes here; the producer works
// in UTF-8 either way.

bool set_environment(const char*name,const char*value){return setenv(name,value,1)==0;}
bool file_absent(const std::filesystem::path&p){struct stat st{};return stat(p.c_str(),&st)!=0&&errno==ENOENT;}
bool file_present(const std::filesystem::path&p){struct stat st{};return stat(p.c_str(),&st)==0;}
bool remove_file(const std::filesystem::path&p){return unlink(p.c_str())==0;}
// MemAvailable, the figure llama_dsv41_host_memory admits against: reclaimable
// page cache counts as available, which is what the physical-RAM floor means.
bool available_ram(uint64_t&bytes){llama_dsv41_host_memory::state s;if(!llama_dsv41_host_memory::read(s))return false;bytes=s.available;return true;}

uint64_t ram_admission(uint64_t admitted){uint64_t bytes=0;
    require(available_ram(bytes),"Cannot inspect available physical RAM");
    require(bytes>=add(admitted,gib),"Producer job host plan exceeds available physicalRAM minus1GiB");return bytes;}

// Process-wide cancellation. SIGINT and SIGTERM (the Ctrl+C/Ctrl+Break/Ctrl+close
// trio this replaces) only latch the flag: the producer stops at its next bounded
// operation and keeps unfinished output.
void interrupt_signal(int){interrupted.store(true);}
void install_cancellation(){
    struct sigaction action{};
    action.sa_handler=interrupt_signal;
    sigemptyset(&action.sa_mask);
    action.sa_flags=0;
    require(sigaction(SIGINT,&action,nullptr)==0&&sigaction(SIGTERM,&action,nullptr)==0,"Cannot install producer cancellation handler");
}

// LLAMADECODE: an external graceful abort. The caller's RAM watch creates this
// file; the engine then stops at its next stage boundary and tears down normally.
bool abort_file_present(){
    static const char*name=std::getenv("LLAMADECODE_ABORT_FILE");
    if(!name||!*name)return false;
    static std::atomic<int64_t>last{0};const int64_t now=std::chrono::duration_cast<std::chrono::milliseconds>(clock_type::now().time_since_epoch()).count();
    if(now-last.load()<250)return false;last.store(now);
    return file_present(producer_text::native_path(name));
}

// Both runtime guards publish their settings and latched stop verdicts.
json host_memory_runtime(){namespace hm=llama_dsv41_host_memory;
    return json{{"rule","stop below runtime_min_free_gib (0 disables) or when pagefile growth exceeds limit_bytes"},
        {"runtime_min_free_gib",double(hm::runtime_min_free_bytes)/double(gib)},{"floor_stop",hm::floor_stop.load()},
        {"memory_query_failed",hm::memory_query_failed.load()},
        {"limit_bytes",hm::paging_limit},{"pagefile_growth_peak_bytes",hm::paging_growth_peak.load()},{"low_memory_events",hm::low_memory_events.load()},{"paging_stop",hm::paging_stop.load()}};}

// Exclusive-new writer for the bounded diagnostic outputs a session operation
// publishes (KV samples, forced decode logits). O_EXCL is the exclusive-new
// admission; Linux cannot deny another process opening the file, so the producer
// simply holds the only writer while it owns it.
class exclusive_writer {
    producer_wire::handle file=-1;
public:
    ~exclusive_writer(){if(held())producer_wire::release(file);}
    bool held()const{return file!=producer_wire::unarmed();}
    void create(const std::filesystem::path&p){
        file=::open(p.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0644);
    }
    bool write(const void*src,size_t bytes){return producer_wire::write_all(file,src,bytes);}
    bool release(){
        if(!held())return false;
        const bool done=::close(file)==0;
        file=producer_wire::unarmed();return done;
    }
};

// Process-lifetime dynamic libraries: the decode engine and NVML. Neither is
// unloaded, because ds4 code may still be running in a CUDA callback it started.
using library=void*;
inline library open_library(const char*name,int flags){return dlopen(name,flags);}
inline void*library_symbol(library module,const char*name){return module?dlsym(module,name):nullptr;}
inline std::string library_error(){const char*why=dlerror();return why?why:"no error";}
inline library open_nvml(){return open_library("libnvidia-ml.so.1",RTLD_NOW|RTLD_LOCAL);}
inline std::filesystem::path decode_library_path(){return executable().parent_path()/std::string("ds4-decode.so");}
inline library open_decode_library(){return open_library(decode_library_path().c_str(),RTLD_NOW|RTLD_LOCAL);}
constexpr const char*decode_library_name="ds4-decode.so";

// Physical device memory from NVML (the driver's own count; the CUDA free count
// double-counts imported device banks). Selected by the job's GPU UUID.
bool nvml_free_bytes(uint64_t&free_bytes,uint64_t&total_bytes){
    free_bytes=total_bytes=0;
    static library nvml=open_nvml();
    if(!nvml)return false;
    struct memory{unsigned long long total,free,used;};
    using init_t=int(*)();using by_uuid_t=int(*)(const char*,void**);using info_t=int(*)(void*,memory*);
    static auto init=reinterpret_cast<init_t>(library_symbol(nvml,"nvmlInit_v2"));
    static auto by_uuid=reinterpret_cast<by_uuid_t>(library_symbol(nvml,"nvmlDeviceGetHandleByUUID"));
    static auto info=reinterpret_cast<info_t>(library_symbol(nvml,"nvmlDeviceGetMemoryInfo"));
    static const int ready=init?init():-1;
    const char*uuid=std::getenv("CUDA_VISIBLE_DEVICES");
    if(ready!=0||!by_uuid||!info||!uuid||std::strncmp(uuid,"GPU-",4))return false;
    void*device=nullptr;memory m{};
    if(by_uuid(uuid,&device)!=0||info(device,&m)!=0)return false;
    free_bytes=m.free;total_bytes=m.total;return true;
}
// The device memory this process holds. The Windows producer asked DXGI for its
// WDDM local-segment budget and current usage, the authoritative no-demotion
// figure; Linux has no per-process segment budget, so this reports the device's
// total and in-use bytes. Both are diagnostics beside the NVML free count, which
// is what actually bounds an L2 offer.
void local_video_memory(uint64_t&budget,uint64_t&usage){
    budget=usage=0;
    uint64_t free_bytes=0,total=0;
    if(!nvml_free_bytes(free_bytes,total))return;
    budget=total;usage=total>free_bytes?total-free_bytes:0;
}

} // namespace
