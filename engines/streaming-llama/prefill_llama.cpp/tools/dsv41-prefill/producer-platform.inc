// Producer platform layer. Included by tools/dsv41-prefill/main.cpp after the
// producer's own policy helpers (require, stop, the bounded constants) and
// before live-session.inc, so both see these names in the same namespace.
//
// The Windows branch is the code the producer ran before this fork went
// Linux-only, kept so the same source still builds there. The POSIX branch is
// the port; the substitutions are the ones docs/linux-port-plan.md Phase 2
// lists:
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
// Two Windows guarantees have no POSIX twin and are deliberately not faked:
// share modes (Linux cannot deny another process opening the same file), and
// fixed-vs-removable media (no statfs indicator). Identity rechecks before and
// after every file operation are what stands in for the first; network and
// RAM-backed volumes are refused by type.
#include "boundary-capture.h"        // dsv41_boundary::sha256_native
#include "llama-dsv41-host-memory.h"   // the RAM figure available_ram reports
#include "producer-platform.h"         // argument, path_text, native_path
#include "producer-wire.h"            // the session handle exclusive_writer uses

namespace {
using producer_text::argument;using producer_text::native_path;using producer_text::path_text;

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <dxgi1_4.h>
#else
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
#endif

// Incremental SHA-256 of the bytes the producer writes or binds. update() is
// bounded to one chunk; finish() finalizes and the object is done.
#if defined(_WIN32)
struct hash {
    BCRYPT_ALG_HANDLE algorithm=nullptr;BCRYPT_HASH_HANDLE handle=nullptr;std::vector<uint8_t>object;
    hash(){
        require(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)>=0,"Cannot open native SHA256");
        DWORD bytes=0,n=0;require(BCryptGetProperty(algorithm,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&bytes),sizeof(bytes),&n,0)>=0&&bytes<=65536,"Invalid SHA workspace");
        object.resize(bytes);require(BCryptCreateHash(algorithm,&handle,object.data(),bytes,nullptr,0,0)>=0,"Cannot create native SHA256");
    }
    ~hash(){if(handle)BCryptDestroyHash(handle);if(algorithm)BCryptCloseAlgorithmProvider(algorithm,0);}
    void update(const void*data,size_t bytes){require(bytes<=chunk&&BCryptHashData(handle,(PUCHAR)data,ULONG(bytes),0)>=0,"Native SHA256 update failed");}
    std::string finish(){std::array<uint8_t,32>out{};require(BCryptFinishHash(handle,out.data(),ULONG(out.size()),0)>=0,"Native SHA256 finish failed");
        std::ostringstream text;text<<std::hex<<std::setfill('0');for(auto b:out)text<<std::setw(2)<<unsigned(b);return text.str();}
};
#else
// The same digest as the Windows CNG path, in the portable state the boundary
// capture diagnostics already carry on this platform.
struct hash {
    dsv41_boundary::sha256_native digest;
    void update(const void*data,size_t bytes){require(bytes<=chunk,"Native SHA256 update failed");digest.update(data,bytes);}
    std::string finish(){return digest.hex();}
};
#endif

// Physical identity of a held file: the volume it lives on, its inode or file
// id, its length, and its modification time in Windows FILETIME ticks (the unit
// the receipts and the weight lease compare, so both platforms convert).
struct identity {uint64_t volume=0,size=0,mtime=0;std::array<uint8_t,16>id{};};
std::string id_hex(const std::array<uint8_t,16>&id){std::ostringstream s;s<<std::hex<<std::setfill('0');for(auto b:id)s<<std::setw(2)<<unsigned(b);return s.str();}
json describe(const identity&id){return {{"volume_serial_u64",id.volume},{"file_id_le_hex",id_hex(id.id)},{"bytes",id.size},{"mtime_filetime_ticks",id.mtime}};}
bool same(const identity&a,const identity&b){return a.volume==b.volume&&a.id==b.id&&a.size==b.size&&a.mtime==b.mtime;}

#if defined(_WIN32)
identity inspect(HANDLE file){
    FILE_ID_INFO id{};BY_HANDLE_FILE_INFORMATION info{};LARGE_INTEGER size{};
    require(GetFileInformationByHandleEx(file,FileIdInfo,&id,sizeof(id))&&GetFileInformationByHandle(file,&info)&&GetFileSizeEx(file,&size),"Cannot inspect native file identity");
    require(!(info.dwFileAttributes&(FILE_ATTRIBUTE_REPARSE_POINT|FILE_ATTRIBUTE_DIRECTORY|FILE_ATTRIBUTE_SPARSE_FILE|FILE_ATTRIBUTE_COMPRESSED))&&info.nNumberOfLinks==1&&size.QuadPart>=0,
        "Producer requires a physical single-link ordinary file");
    identity out;out.volume=id.VolumeSerialNumber;out.size=uint64_t(size.QuadPart);std::memcpy(out.id.data(),id.FileId.Identifier,16);
    out.mtime=(uint64_t(info.ftLastWriteTime.dwHighDateTime)<<32)|info.ftLastWriteTime.dwLowDateTime;return out;
}
#else
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
#endif

// No unlink, replacement, truncation or cleanup operation exists here. The
// sole rename publishes our new flushed pending receipt without replacing.
// Held physical directory handles deny DELETE on Windows and pin the ancestor
// chain here; before each file action those identities are rebound to the
// current named chain as well.
class native_file {
    std::filesystem::path path;
#if defined(_WIN32)
    std::vector<HANDLE>parents;
    std::vector<std::filesystem::path>names;
    HANDLE file=INVALID_HANDLE_VALUE;
#else
    std::vector<int>parents;
    std::vector<std::filesystem::path>names;
    int file=-1;
#endif
    identity owned{};
    uint64_t limit=0,pending=0,disk_floor=reserve;
    bool output=false,can_publish=false,flushed=false;

    void parents_ok()const{
        for(size_t i=0;i<parents.size();++i){
#if defined(_WIN32)
            BY_HANDLE_FILE_INFORMATION held{},named{};
            require(GetFileInformationByHandle(parents[i],&held)&&
                (held.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)&&!(held.dwFileAttributes&FILE_ATTRIBUTE_REPARSE_POINT),"Held output ancestor changed");
            HANDLE probe=CreateFileW(names[i].c_str(),FILE_READ_ATTRIBUTES,FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,OPEN_EXISTING,
                FILE_FLAG_BACKUP_SEMANTICS|FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
            const bool valid=probe!=INVALID_HANDLE_VALUE&&GetFileInformationByHandle(probe,&named)&&
                (named.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)&&!(named.dwFileAttributes&FILE_ATTRIBUTE_REPARSE_POINT)&&
                held.dwVolumeSerialNumber==named.dwVolumeSerialNumber&&held.nFileIndexHigh==named.nFileIndexHigh&&held.nFileIndexLow==named.nFileIndexLow;
            if(probe!=INVALID_HANDLE_VALUE)CloseHandle(probe);require(valid,"Named output ancestor identity changed");
#else
            struct stat held{},named{};
            require(fstat(parents[i],&held)==0&&S_ISDIR(held.st_mode),"Held output ancestor changed");
            // O_NOFOLLOW refuses a symlink where Windows refused a reparse point.
            const int probe=::open(names[i].c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
            const bool valid=probe>=0&&fstat(probe,&named)==0&&S_ISDIR(named.st_mode)&&
                held.st_dev==named.st_dev&&held.st_ino==named.st_ino;
            if(probe>=0)close(probe);
            require(valid,"Named output ancestor identity changed");
#endif
        }
    }

public:
    native_file()=default;
    ~native_file(){if(file!=closed_marker())close_held();for(auto h:parents)release(h);}
    native_file(const native_file&)=delete;native_file&operator=(const native_file&)=delete;

    // The selected volume's identity, and the rule that it is a local disk.
    static uint64_t volume(const std::filesystem::path&p){
        require(p.is_absolute()&&p==p.lexically_normal()&&p.has_filename(),"Producer paths must be absolute and normalized");
#if defined(_WIN32)
        require(GetDriveTypeW(p.root_path().c_str())==DRIVE_FIXED,"Producer requires a fixed local volume");
        wchar_t device[32768]{};require(QueryDosDeviceW(p.root_name().c_str(),device,32768)&&std::wstring(device).find(L"\\Device\\HarddiskVolume")==0,
            "Producer requires a physical volume, not UNC/SUBST");
        HANDLE root=CreateFileW(p.root_path().c_str(),FILE_READ_ATTRIBUTES,FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS|FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
        require(root!=INVALID_HANDLE_VALUE,"Cannot inspect producer volume");FILE_ID_INFO info{};
        const bool valid=GetFileInformationByHandleEx(root,FileIdInfo,&info,sizeof(info))!=0;CloseHandle(root);require(valid,"Cannot identify producer volume");return info.VolumeSerialNumber;
#else
        struct statfs probe{};
        // The directory, not the file: volume() is also called for an output
        // path that does not exist yet.
        require(statfs(p.parent_path().c_str(),&probe)==0,"Cannot inspect producer volume");
        // Windows refused anything that is not DRIVE_FIXED. Linux has no fixed/
        // removable indicator, so the remote and RAM-backed types are refused by
        // their magic numbers instead; an unknown local type is accepted.
        const long type=long(probe.f_type);
        const long refused[]={NFS_SUPER_MAGIC,FUSE_SUPER_MAGIC,0x5346544D/*SMB*/,0xfe534d42/*SMB2*/,0xff53ff53/*CIFS*/,
            0x53464146/*AFS*/,0x010209/*9P*/,0x7461636f/*OCFS2*/,0x01161970/*GFS2*/,0x564c/*NCP*/,
            TMPFS_MAGIC,RAMFS_MAGIC,0x516194e9/*devtmpfs*/};
        for(const long magic:refused)require(type!=magic,"Producer requires a fixed local volume, not a network or RAM-backed one");
        struct ::stat root{};
        require(::stat(p.parent_path().c_str(),&root)==0&&S_ISDIR(root.st_mode),"Cannot identify producer volume");
        return uint64_t(root.st_dev);
#endif
    }

    void open(const std::filesystem::path&p,bool create,uint64_t expected,uint64_t outstanding=0,bool publishable=false,uint64_t output_floor=reserve){
#if defined(_WIN32)
        require(file==INVALID_HANDLE_VALUE&&parents.empty(),"Native producer file is single-use");
#else
        require(file==-1&&parents.empty(),"Native producer file is single-use");
#endif
        path=p;volume(path);
        require(path_text(path.filename()).find(':')==std::string::npos,"Alternate data streams are forbidden");
        std::vector<std::filesystem::path>chain;for(auto d=path.parent_path();;d=d.parent_path()){chain.push_back(d);if(d==d.root_path())break;}
        std::reverse(chain.begin(),chain.end());
        for(const auto&d:chain){
#if defined(_WIN32)
            HANDLE h=CreateFileW(d.c_str(),FILE_READ_ATTRIBUTES,FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,OPEN_EXISTING,
                FILE_FLAG_BACKUP_SEMANTICS|FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
            require(h!=INVALID_HANDLE_VALUE,"Cannot lease physical producer ancestor");
#else
            const int h=::open(d.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
            require(h>=0,"Cannot lease physical producer ancestor");
#endif
            parents.push_back(h);names.push_back(d);parents_ok();
        }
        limit=expected;pending=outstanding;disk_floor=output_floor;output=create;can_publish=publishable;parents_ok();if(create)budget();parents_ok();
#if defined(_WIN32)
        file=CreateFileW(path.c_str(),create?(GENERIC_READ|GENERIC_WRITE|(publishable?DELETE:0)):GENERIC_READ,FILE_SHARE_READ,nullptr,create?CREATE_NEW:OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL|FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
        require(file!=INVALID_HANDLE_VALUE,"Cannot open exclusive-new output or read-only physical input");
#else
        // No share-mode twin: O_EXCL keeps the exclusive-new admission, and the
        // identity rechecks refuse a file that another process replaced.
        const int flags=(create?(O_RDWR|O_CREAT|O_EXCL):(O_RDONLY))|O_NOFOLLOW|O_CLOEXEC;
        file=::open(path.c_str(),flags,0644);
        require(file>=0,"Cannot open exclusive-new output or read-only physical input");
#endif
        owned=inspect(file);
        require(create?owned.size==0:owned.size<=limit,"Producer file exceeds its bounded byte limit");check();
    }
    void check()const{parents_ok();require(same(owned,inspect(file)),"Native producer file identity/size/mtime changed");}
    void budget()const{
        parents_ok();
#if defined(_WIN32)
        ULARGE_INTEGER free{};require(GetDiskFreeSpaceExW(path.parent_path().c_str(),&free,nullptr,nullptr),"Cannot inspect actual output-volume free bytes");
        require(ds4perf_output_budget::fits(free.QuadPart,limit,owned.size,pending,disk_floor),
            "Producer output/carry/other pending bytes exceed actual-volume budget plus selected disk reserve");
#else
        struct statvfs probe{};
        require(statvfs(path.parent_path().c_str(),&probe)==0,"Cannot inspect actual output-volume free bytes");
        const uint64_t free_bytes=uint64_t(probe.f_bavail)*uint64_t(probe.f_frsize);
        require(ds4perf_output_budget::fits(free_bytes,limit,owned.size,pending,disk_floor),
            "Producer output/carry/other pending bytes exceed actual-volume budget plus selected disk reserve");
#endif
    }
    void outstanding(uint64_t bytes){pending=bytes;}
    identity current()const{check();return owned;}
    size_t read(void*dst,size_t bytes){
        require(!output&&bytes<=chunk,"Invalid bounded input read");check();
#if defined(_WIN32)
        DWORD got=0;require(ReadFile(file,dst,DWORD(bytes),&got,nullptr),"Producer input read failed");check();return got;
#else
        const ssize_t got=::read(file,dst,bytes);
        require(got>=0,"Producer input read failed");check();return size_t(got);
#endif
    }
    void write(const void*src,size_t bytes){
        require(output&&bytes<=chunk&&owned.size<=limit&&bytes<=limit-owned.size,"Producer output exceeds planned bounded size");
        check();budget();flushed=false;
#if defined(_WIN32)
        DWORD wrote=0;require(WriteFile(file,src,DWORD(bytes),&wrote,nullptr)&&wrote==bytes,"Native output write failed; incomplete file retained");
#else
        const unsigned char*p=static_cast<const unsigned char*>(src);size_t left=bytes;
        while(left){const ssize_t sent=::write(file,p,left);require(sent>0,"Native output write failed; incomplete file retained");p+=size_t(sent);left-=size_t(sent);}
#endif
        const auto after=inspect(file);require(after.volume==owned.volume&&after.id==owned.id&&after.size==owned.size+bytes,"Native output ownership or length changed");owned=after;check();
    }
    void flush(){
        check();
#if defined(_WIN32)
        require(FlushFileBuffers(file),"Native output flush failed; output is not certified");
#else
        require(fsync(file)==0,"Native output flush failed; output is not certified");
#endif
        owned=inspect(file);check();flushed=true;
    }
    void finalize_read_only(const std::string&expected_sha){
        require(output&&flushed,"Payload must be flushed before closing its writer");check();const auto before=owned;
#if defined(_WIN32)
        require(CloseHandle(file),"Cannot close completed payload writer");file=INVALID_HANDLE_VALUE;parents_ok();
        file=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_OPEN_REPARSE_POINT,nullptr);
        require(file!=INVALID_HANDLE_VALUE,"Cannot acquire completed payload readonly lease");
#else
        require(close(file)==0,"Cannot close completed payload writer");file=-1;parents_ok();
        const int readonly=::open(path.c_str(),O_RDONLY|O_NOFOLLOW|O_CLOEXEC);
        require(readonly>=0,"Cannot acquire completed payload readonly lease");file=readonly;
#endif
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
#if defined(_WIN32)
        const auto name=destination.wstring();
        // SetFileInformationByHandle reads the NUL-terminated FileName path.
        // Keep FileNameLength exclusive of NUL, but include NUL in the buffer.
        const size_t bytes=offsetof(FILE_RENAME_INFO,FileName)+(name.size()+1)*sizeof(wchar_t);
        require(bytes<=65536,"Receipt publish path exceeds bounded metadata");std::vector<uint8_t>buffer(bytes,0);
        auto*rename=reinterpret_cast<FILE_RENAME_INFO*>(buffer.data());rename->ReplaceIfExists=FALSE;rename->RootDirectory=nullptr;
        rename->FileNameLength=DWORD(name.size()*sizeof(wchar_t));std::memcpy(rename->FileName,name.data(),rename->FileNameLength);
        require(SetFileInformationByHandle(file,FileRenameInfo,rename,DWORD(bytes)),"No-replace receipt publication failed; pending file retained");
#else
        // RENAME_NOREPLACE fails with EEXIST on an occupied target instead of
        // publishing over it, which is what ReplaceIfExists=FALSE means.
        require(renameat2(AT_FDCWD,path.c_str(),AT_FDCWD,destination.c_str(),RENAME_NOREPLACE)==0,
            "No-replace receipt publication failed; pending file retained");
#endif
        // No fallible operation follows the atomic publication.
        path.swap(destination);
    }

private:
#if defined(_WIN32)
    static HANDLE closed_marker(){return INVALID_HANDLE_VALUE;}
    void close_held(){CloseHandle(file);file=INVALID_HANDLE_VALUE;}
    static void release(HANDLE h){CloseHandle(h);}
#else
    static int closed_marker(){return -1;}
    void close_held(){close(file);file=-1;}
    static void release(int h){close(h);}
#endif
};

struct contents {std::vector<uint8_t>bytes;std::string sha;identity id;};
contents read_small(const std::filesystem::path&path,uint64_t maximum,bool retain=true){
    native_file file;file.open(path,false,maximum);contents out;out.id=file.current();hash digest;
    std::vector<uint8_t>buffer(chunk);uint64_t read=0;
    for(;;){stop();const auto count=file.read(buffer.data(),buffer.size());if(!count)break;read+=count;require(read<=maximum,"Input grew beyond bounded read admission");
        digest.update(buffer.data(),count);if(retain)out.bytes.insert(out.bytes.end(),buffer.begin(),buffer.begin()+count);}
    require(read==out.id.size,"Input byte count changed while reading");out.sha=digest.finish();return out;
}

#if defined(_WIN32)
std::filesystem::path executable(){wchar_t name[32768]{};const auto n=GetModuleFileNameW(nullptr,name,32768);require(n&&n<32768,"Cannot identify actual producer binary");return std::filesystem::path(std::wstring(name,n));}
uint64_t process_id(){return uint64_t(GetCurrentProcessId());}
bool set_environment(const char*name,const char*value){return _putenv_s(name,value)==0;}
bool file_absent(const std::filesystem::path&p){return GetFileAttributesW(p.c_str())==INVALID_FILE_ATTRIBUTES&&GetLastError()==ERROR_FILE_NOT_FOUND;}
bool file_present(const std::filesystem::path&p){return GetFileAttributesW(p.c_str())!=INVALID_FILE_ATTRIBUTES;}
bool remove_file(const std::filesystem::path&p){return DeleteFileW(p.c_str())!=0;}
bool available_ram(uint64_t&bytes){MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);if(!GlobalMemoryStatusEx(&memory))return false;bytes=memory.ullAvailPhys;return true;}
#else
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
// page cache counts as available here exactly as ullAvailPhys does on Windows.
bool available_ram(uint64_t&bytes){llama_dsv41_host_memory::state s;if(!llama_dsv41_host_memory::read(s))return false;bytes=s.available;return true;}
#endif

uint64_t ram_admission(uint64_t admitted){uint64_t bytes=0;
    require(available_ram(bytes),"Cannot inspect available physical RAM");
    require(bytes>=add(admitted,gib),"Producer job host plan exceeds available physicalRAM minus1GiB");return bytes;}

// Process-wide cancellation: Ctrl+C/Ctrl+Break/Ctrl+close on Windows, SIGINT
// and SIGTERM here. Both only latch the flag; the producer stops at its next
// bounded operation and keeps unfinished output.
#if defined(_WIN32)
BOOL WINAPI interrupt(DWORD kind){if(kind==CTRL_C_EVENT||kind==CTRL_BREAK_EVENT||kind==CTRL_CLOSE_EVENT){interrupted.store(true);return TRUE;}return FALSE;}
void install_cancellation(){require(SetConsoleCtrlHandler(interrupt,TRUE),"Cannot install producer cancellation handler");}
#else
void interrupt_signal(int){interrupted.store(true);}
void install_cancellation(){
    struct sigaction action{};
    action.sa_handler=interrupt_signal;
    sigemptyset(&action.sa_mask);
    action.sa_flags=0;
    require(sigaction(SIGINT,&action,nullptr)==0&&sigaction(SIGTERM,&action,nullptr)==0,"Cannot install producer cancellation handler");
}
#endif

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
// publishes (KV samples, forced decode logits). Windows denied every share mode
// on the handle; here O_EXCL is the exclusive-new admission and nothing else
// holds the file while this writer owns it.
class exclusive_writer {
#if defined(_WIN32)
    producer_wire::handle file=nullptr;
#else
    producer_wire::handle file=-1;
#endif
public:
    ~exclusive_writer(){if(held())producer_wire::release(file);}
    bool held()const{return file!=producer_wire::unarmed();}
    void create(const std::filesystem::path&p){
#if defined(_WIN32)
        file=CreateFileW(p.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
#else
        file=::open(p.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0644);
#endif
    }
    bool write(const void*src,size_t bytes){return producer_wire::write_all(file,src,bytes);}
    bool release(){
        if(!held())return false;
#if defined(_WIN32)
        const bool done=CloseHandle(file)!=0;
#else
        const bool done=::close(file)==0;
#endif
        file=producer_wire::unarmed();return done;
    }
};

// Process-lifetime dynamic libraries: the decode engine and NVML. Neither is
// unloaded, because ds4 code may still be running in a CUDA callback it started.
#if defined(_WIN32)
using library=HMODULE;
inline library open_library(const wchar_t*name,DWORD flags){return LoadLibraryExW(name,nullptr,flags);}
inline void*library_symbol(library module,const char*name){return module?GetProcAddress(module,name):nullptr;}
inline std::string library_error(){return std::to_string(GetLastError());}
inline library open_nvml(){return open_library(L"nvml.dll",LOAD_LIBRARY_SEARCH_SYSTEM32);}
// The decode library beside this executable.
inline library open_decode_library(){
    return open_library((executable().parent_path()/L"ds4-decode.dll").c_str(),
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
}
constexpr const char*decode_library_name="ds4-decode.dll";
#else
using library=void*;
inline library open_library(const char*name,int flags){return dlopen(name,flags);}
inline void*library_symbol(library module,const char*name){return module?dlsym(module,name):nullptr;}
inline std::string library_error(){const char*why=dlerror();return why?why:"no error";}
inline library open_nvml(){return open_library("libnvidia-ml.so.1",RTLD_NOW|RTLD_LOCAL);}
inline std::filesystem::path decode_library_path(){return executable().parent_path()/std::string("ds4-decode.so");}
inline library open_decode_library(){return open_library(decode_library_path().c_str(),RTLD_NOW|RTLD_LOCAL);}
constexpr const char*decode_library_name="ds4-decode.so";
#endif

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
// The room this process may hold on the local segment. Windows asks DXGI for the
// WDDM budget and its current usage, which is the authoritative no-demotion
// figure. Linux has no process segment budget, so the device's own total and
// in-use bytes stand in; both are diagnostics beside the NVML cap that actually
// bounds an offer.
void local_video_memory(uint64_t&budget,uint64_t&usage){
    budget=usage=0;
#if defined(_WIN32)
    IDXGIFactory4*factory=nullptr;
    if(FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4),reinterpret_cast<void**>(&factory)))||!factory)return;
    uint64_t best=0;
    for(UINT i=0;;++i){IDXGIAdapter1*a=nullptr;if(factory->EnumAdapters1(i,&a)==DXGI_ERROR_NOT_FOUND)break;
        DXGI_ADAPTER_DESC1 d{};a->GetDesc1(&d);IDXGIAdapter3*a3=nullptr;
        if(d.DedicatedVideoMemory>best&&SUCCEEDED(a->QueryInterface(__uuidof(IDXGIFactory3),reinterpret_cast<void**>(&a3)))){
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if(SUCCEEDED(a3->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&info))){best=d.DedicatedVideoMemory;budget=info.Budget;usage=info.CurrentUsage;}
            a3->Release();}
        a->Release();}
    factory->Release();
#else
    uint64_t free_bytes=0,total=0;
    if(!nvml_free_bytes(free_bytes,total))return;
    budget=total;usage=total>free_bytes?total-free_bytes:0;
#endif
}
// Usage of the non-local (shared) segment, which is what a WDDM overflow shows
// up in. No Linux twin: there is no shared segment to watch, so 0.
uint64_t shared_video_memory_usage(){
#if defined(_WIN32)
    IDXGIFactory4*fa=nullptr;uint64_t shared=0,best=0;
    if(FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4),reinterpret_cast<void**>(&fa)))||!fa)return 0;
    for(UINT i=0;;++i){IDXGIAdapter1*a=nullptr;if(fa->EnumAdapters1(i,&a)==DXGI_ERROR_NOT_FOUND)break;DXGI_ADAPTER_DESC1 d{};a->GetDesc1(&d);IDXGIAdapter3*a3=nullptr;
        if(d.DedicatedVideoMemory>best&&SUCCEEDED(a->QueryInterface(__uuidof(IDXGIFactory3),reinterpret_cast<void**>(&a3)))){DXGI_QUERY_VIDEO_MEMORY_INFO nl{};
            if(SUCCEEDED(a3->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,&nl))){best=d.DedicatedVideoMemory;shared=nl.CurrentUsage;}a3->Release();}
        a->Release();}
    fa->Release();return shared;
#else
    return 0;
#endif
}

} // namespace
