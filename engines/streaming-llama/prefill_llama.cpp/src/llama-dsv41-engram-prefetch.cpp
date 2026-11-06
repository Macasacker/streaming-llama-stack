// Linux: the Engram row loader reads its extents through io_uring, one read
// queue per ring, and keeps the serial `pread` baseline beside it
// (LLAMADECODE_ENGRAM_IO_URING=0) so the ring's gain can be measured against
// the same rows. Ring declarations are isolated to this translation unit, as
// they are in the weight fetcher's reader.
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#include <sys/eventfd.h>
#include <sys/stat.h>
#include <sys/types.h>
#if __has_include(<liburing.h>)
#  define ENGRAM_HAS_IO_URING 1
#  include <liburing.h>
#endif
#include "llama-dsv41-layer-log.h"
#include "llama-dsv41-engram-prefetch.h"
#include "llama-dsv41-engram-plan.h"
#include "llama-dsv41-engram-early-decision.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
// The early multi-ring loader serves every Engram load (all EARLY arms). =0:
// no early start, and the weight-stream marker is never set.
static std::atomic<bool> llama_dsv41_engram_weight_pass{false};
static uint32_t engram_read_alignment(){
    static const uint32_t alignment=[]{
        const char*e=std::getenv("LLAMADECODE_ENGRAM_READ_ALIGN");
        if(!e||!*e||!std::strcmp(e,"4096"))return 4096u;
        if(!std::strcmp(e,"512"))return 512u;
        throw std::invalid_argument("LLAMADECODE_ENGRAM_READ_ALIGN must be 512 or 4096");
    }();return alignment;
}
static llama_dsv41_engram_early::mode llama_dsv41_engram_early_mode(){
    static const auto m=llama_dsv41_engram_early::parse_mode(std::getenv("LLAMADECODE_ENGRAM_EARLY"));return m;
}
bool llama_dsv41_engram_whole_prompt(){
    static const bool v=[]{const char*e=std::getenv("LLAMADECODE_ENGRAM_WHOLE_PROMPT");
        if(!e||!*e||!std::strcmp(e,"1"))return true;if(!std::strcmp(e,"0"))return false;
        throw std::invalid_argument("LLAMADECODE_ENGRAM_WHOLE_PROMPT must be 0 or 1");}();return v;
}
bool llama_dsv41_engram_early_enabled(){return llama_dsv41_engram_early_mode()!=llama_dsv41_engram_early::mode::off;}
bool llama_dsv41_engram_early_auto(){return llama_dsv41_engram_early_mode()==llama_dsv41_engram_early::mode::automatic;}
bool llama_dsv41_engram_early_pause(){return llama_dsv41_engram_early_mode()==llama_dsv41_engram_early::mode::pause;}
double llama_dsv41_engram_early_iops(){
    static const double v=llama_dsv41_engram_early::parse_iops(std::getenv("LLAMADECODE_ENGRAM_IOPS"));return v;
}
void llama_dsv41_engram_weight_stream(bool active){llama_dsv41_engram_weight_pass.store(active,std::memory_order_relaxed);}
namespace {
// LLAMADECODE_ENGRAM_IO_URING=0 selects the serial `pread` baseline; every
// other value, and an unset environment, keeps the ring reader. Same rule as
// the weight fetcher's LLAMA_WF_IO_URING, and the same reason: the baseline is
// what the ring's gain is measured against. Only read when liburing is compiled
// in - without it the loader has no ring to decline.
#ifdef ENGRAM_HAS_IO_URING
bool engram_uring_wanted(){
    static const bool v=[]{const char*e=std::getenv("LLAMADECODE_ENGRAM_IO_URING");
        if(!e||!*e||!std::strcmp(e,"1"))return true;
        if(!std::strcmp(e,"0"))return false;
        throw std::invalid_argument("LLAMADECODE_ENGRAM_IO_URING must be 0 or 1");}();
    return v;
}
#endif
using engram_clock=std::chrono::steady_clock;
void engram_require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
double engram_ms_since(engram_clock::time_point t){
    return std::chrono::duration<double,std::milli>(engram_clock::now()-t).count();
}
// A cancel request is itself a submitted entry, so its CQE is accounted like
// any other; this user_data marks it as the cancel's own completion rather than
// a read's, so nothing treats it as a row.
constexpr uint64_t engram_cancel_nonce=~uint64_t(0);

struct read_queue;
// Bounded wait for the next completion on any queue sharing one eventfd. The
// caller flushes every submission queue first, so nothing here relies on a wait
// also being a submit (the rule the fetcher's reader learned in chunk 3.1f).
// The eventfd counter is consumed before polling, which can swallow the wake of
// a completion whose CQE is already queued but not yet peeked; each queue's own
// CQ ring is therefore checked before the poll, or it runs out its whole
// timeout and the load loses 100 ms.
void wait_for_completion(int event_fd,const std::vector<const read_queue*>& queues);

// One read queue: an io_uring ring when the kernel offers one, otherwise the
// serial `pread` baseline. The Windows loader used one IoRing per queue with
// the file handle and the slot arena registered through ring operations;
// here registration is a synchronous syscall, optional, and the arena is named
// per request instead of registered. `in_sq` counts entries the kernel has not
// seen, `in_kernel` reads whose completion is still owed.
struct read_queue {
    struct completion {uint64_t id=0;int res=0;};
    int fd=-1;
    bool uring=false;
#ifdef ENGRAM_HAS_IO_URING
    struct io_uring ring{};
    int fixed=-1;                 // registered file index, -1 when unregistered
#endif
    uint32_t in_kernel=0;
    uint32_t in_sq=0;
    std::deque<completion> ready; // the baseline's own completions
    uint32_t outstanding()const{return in_kernel+in_sq+uint32_t(ready.size());}
    // A ring that cannot start, or cannot report completions to the shared
    // eventfd, leaves this queue on the baseline; the loader still reads every
    // row, it just reads one at a time.
    void open(unsigned entries,int event_fd){
#ifdef ENGRAM_HAS_IO_URING
        if(!engram_uring_wanted())return;
        const int init=io_uring_queue_init(entries,&ring,0);
        if(init!=0){
            llama_dsv41_layer_log("LLAMADECODE_ENGRAM_IO_URING ring_init_failed rc=%d reader=pread\n",init);
            return;
        }
        const int posted=io_uring_register_eventfd(&ring,event_fd);
        if(posted!=0){
            llama_dsv41_layer_log("LLAMADECODE_ENGRAM_IO_URING eventfd_failed rc=%d reader=pread\n",posted);
            io_uring_queue_exit(&ring);
            return;
        }
        uring=true;
        if(io_uring_register_files(&ring,&fd,1)==0)fixed=0;
#endif
        (void)entries;(void)event_fd;
    }
    void close(){
#ifdef ENGRAM_HAS_IO_URING
        if(uring){io_uring_unregister_files(&ring);io_uring_queue_exit(&ring);uring=false;}
#endif
    }
    // Flush the submission queue. A zero return is a legitimate submit of
    // nothing new; only a negative one is an error. Entries the kernel has not
    // seen stay in the ring, so a short flush loses nothing.
    bool flush(){
#ifdef ENGRAM_HAS_IO_URING
        if(uring){
            const int n=io_uring_submit(&ring);
            if(n<0)return false;
            const uint32_t moved=std::min<uint32_t>(uint32_t(n),in_sq);
            in_sq-=moved;in_kernel+=moved;
        }
#endif
        return true;
    }
    void request(uint64_t id,void* dst,uint32_t bytes,uint64_t offset){
#ifdef ENGRAM_HAS_IO_URING
        if(uring){
            struct io_uring_sqe* sqe=io_uring_get_sqe(&ring);
            engram_require(sqe,"Engram read queue overflowed its submission ring");
            io_uring_prep_read(sqe,fixed>=0?fixed:fd,dst,bytes,offset);
            io_uring_sqe_set_data(sqe,reinterpret_cast<void*>(id));
            if(fixed>=0)sqe->flags|=IOSQE_FIXED_FILE;
            ++in_sq;return;
        }
#endif
        uint32_t done=0;
        while(done<bytes){
            const ssize_t got=::pread(fd,static_cast<uint8_t*>(dst)+done,bytes-done,off_t(offset+done));
            if(got<=0){ready.push_back({id,got<0?-int(errno):-EIO});return;}
            done+=uint32_t(got);
        }
        ready.push_back({id,int(done)});
    }
    // True when a completion was taken. Every submitted entry yields exactly
    // one CQE, so one read (or one cancel request) is retired per CQE; the
    // caller decides whether it accounts for a row.
    bool peek(completion& out){
        if(!ready.empty()){out=ready.front();ready.pop_front();return true;}
#ifdef ENGRAM_HAS_IO_URING
        if(uring){
            struct io_uring_cqe* cqe=nullptr;
            if(io_uring_peek_cqe(&ring,&cqe)!=0)return false;
            out.id=reinterpret_cast<uint64_t>(io_uring_cqe_get_data(cqe));
            out.res=cqe->res;
            io_uring_cqe_seen(&ring,cqe);
            --in_kernel;
            return true;
        }
#endif
        return false;
    }
    // The CancelIoEx the Windows cleanup called on the shared handle: every
    // read still queued on this descriptor, matched by fd rather than nonce.
    void cancel_outstanding(){
#ifdef ENGRAM_HAS_IO_URING
        if(uring&&in_kernel){
            struct io_uring_sqe* sqe=io_uring_get_sqe(&ring);
            if(!sqe)return;
            io_uring_prep_cancel_fd(sqe,fixed>=0?fixed:fd,
                IORING_ASYNC_CANCEL_ALL|(fixed>=0?IORING_ASYNC_CANCEL_FD_FIXED:0));
            io_uring_sqe_set_data(sqe,reinterpret_cast<void*>(engram_cancel_nonce));
            ++in_sq;
        }
#endif
    }
    [[noreturn]] static void unsafe_cleanup(){
        std::fputs("DSV41_ENGRAM_RING_CLEANUP_REFUSED: owned process exits before releasing buffers with unconfirmed IO\n",stderr);
        std::fflush(stderr);_exit(79);std::abort();
    }
    // Every submitted read must be accounted for before its buffer is freed.
    // Unsubmitted SQ entries are abandoned when the ring closes, so only
    // `in_kernel` reads have to drain here.
    void settle(int event_fd){
        if(!flush())unsafe_cleanup();
        cancel_outstanding();
        if(!flush())unsafe_cleanup();
        const auto deadline=engram_clock::now()+std::chrono::seconds(5);
        while(in_kernel){
            completion c;
            if(peek(c))continue;
            if(engram_clock::now()>=deadline)unsafe_cleanup();
            wait(event_fd);
        }
    }
    // True when a completion is already accounted for by this queue: a
    // baseline read in its own list, or a CQE sitting in the ring.
    bool has_completion()const{
        if(!ready.empty())return true;
#ifdef ENGRAM_HAS_IO_URING
        return uring&&io_uring_cq_ready(&ring)!=0;
#else
        return false;
#endif
    }
    void wait(int event_fd){wait_for_completion(event_fd,{this});}
};

void wait_for_completion(int event_fd,const std::vector<const read_queue*>& queues){
    uint64_t cleared=0;
    const ssize_t ignored=::read(event_fd,&cleared,sizeof(cleared));(void)ignored;  // EFD_NONBLOCK: -EAGAIN when nothing posted
    for(const auto* q:queues)if(q->has_completion())return;
    struct pollfd p{event_fd,POLLIN,0};
    engram_require(poll(&p,1,100)>=0,"Engram read queue completion wait failed");
}

// The loader opens its own descriptor rather than sharing the provider's
// lease, exactly as the Windows loader opened its own handles: the two reads
// are different queue depths on different rings, and the identity check below
// is what proves the name still resolves to the file the receipt hashed.
// O_NOFOLLOW refuses a swapped leaf; a swapped ancestor directory resolves to
// a different inode here, which is what the st_dev/st_ino compare refuses, just
// as FILE_ID_INFO did.
int open_verified_row_file(const std::filesystem::path& path,int verified_fd,uint64_t verified_file_bytes){
    engram_require(verified_fd>=0,"Engram loader has no verified model descriptor");
    const int fd=::open(path.c_str(),O_RDONLY|O_NOFOLLOW|O_CLOEXEC);
    engram_require(fd>=0,"Cannot open original Engram file for its own read queues");
    struct stat held{},mine{};
    const bool ok=fstat(verified_fd,&held)==0&&fstat(fd,&mine)==0&&S_ISREG(mine.st_mode)&&mine.st_nlink==1&&
        mine.st_dev==held.st_dev&&mine.st_ino==held.st_ino&&uint64_t(mine.st_size)==verified_file_bytes;
    if(!ok){::close(fd);engram_require(false,"Engram read descriptor differs from the verified model");}
    return fd;
}
uint8_t* allocate_arena(size_t bytes){
    void* p=nullptr;
    engram_require(posix_memalign(&p,4096,bytes)==0&&p,"Cannot allocate admitted Engram read slots");
    return static_cast<uint8_t*>(p);
}
} // namespace
struct llama_dsv41_engram_prompt_cache::implementation {
    static constexpr uint32_t depth=256,page_size=8192,row_bytes=264;
    struct table {std::vector<uint32_t> ids;std::vector<uint8_t> data;};
    std::array<table,2> tables;
    std::array<uint64_t,2> bases{};
    bool attempted=false,prepared=false,load_attempted=false;
    // Tables whose rows are complete: 0, 1 (table 0) or 2 (both). Release
    // stores publish the rows; read() and wait_table() acquire them.
    std::atomic<uint32_t> ready_tables{0};
    std::atomic<bool> failed{false};
    uint64_t host_plan=0;
    // LLAMADECODE_ENGRAM_EARLY only.
    bool early_mode=false;
    uint32_t early_rings=0,early_small=0,early_large=0,early_batch=0;
    // PAIRFIX scan geometry (LLAMADECODE_ENGRAM_SCAN=auto|0|1): a dense row set
    // is read as large sequential extents (gap-tolerant merge) instead of one
    // random 4-8 KiB read per row. Same rows, same bytes copied, same order of
    // table publication; only the read requests differ.
    bool scan=false;uint32_t large_bytes=0;uint64_t merge_gap=0;
    uint64_t geometry_random_requests=0,geometry_scan_requests=0,geometry_scan_bytes=0;
    double geometry_random_s=0,geometry_scan_s=0;
    uint64_t early_arena=0;
    std::shared_ptr<const llama_dsv41_engram_prompt_cache> prior;
    // LLAMADECODE_ENGRAM_CACHE_MB (early loader only).
    std::shared_ptr<llama_dsv41_engram_row_cache> row_cache;
    static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
    static uint64_t add(uint64_t a,uint64_t b){require(a<=UINT64_MAX-b,"Engram cache plan overflow");return a+b;}
    // One queue plus its slot arena. The arena outlives every read: the
    // destructor settles the queue before freeing it, and refuses to free it
    // at all when a read is still owed.
    struct io_scope {
        int fd=-1,event_fd=-1;
        read_queue queue;
        uint8_t* pages=nullptr;
        io_scope(){event_fd=::eventfd(0,EFD_NONBLOCK);require(event_fd>=0,"Cannot create Engram completion event");}
        io_scope(const io_scope&)=delete;io_scope&operator=(const io_scope&)=delete;
        void begin(unsigned entries,size_t arena){
            queue.fd=fd;
            queue.open(entries,event_fd);
            pages=allocate_arena(arena);
        }
        ~io_scope(){
            queue.settle(event_fd);
            queue.close();
            if(pages)std::free(pages);
            if(fd>=0)::close(fd);
            if(event_fd>=0)::close(event_fd);
        }
    };
public:
    void prepare(const llama_dsv41::engram_layout& layout,const int32_t* tokens,uint32_t count,
                 const std::array<uint64_t,2>& offsets,uint64_t admitted_host_bytes,llama_dsv41::engram_history initial,const uint8_t* text_mask){
        require(!attempted&&tokens&&count&&count<=1048576,"Invalid or repeated whole-prompt Engram preparation");attempted=true;
        // Charge both full-capacity ID vectors before allocating them. Dedup
        // does not reduce their capacity, so the final plan retains that charge.
        const uint64_t id_limit=uint64_t(count)*48*sizeof(uint32_t);
        // Async planning retains a copy beside the sweep's borrowed mask.
        const uint64_t mask_bytes=text_mask?uint64_t(count)*2:0;
        require(add(id_limit,4*1024*1024+mask_bytes)<=admitted_host_bytes,"Engram prompt IDs exceed explicit provider host admission");
        for(auto& t:tables)t.ids.reserve(size_t(count)*24);
        llama_dsv41::engram_history history=initial;
        std::vector<llama_dsv41::engram_rows> rows(1024);
        std::string why;
        for(uint32_t start=0;start<count;start+=1024){
            const auto n=std::min(1024u,count-start);
            require(llama_dsv41::engram_hash(layout,history,tokens+start,text_mask?text_mask+start:nullptr,n,rows.data(),why),"Whole-prompt native Engram hash failed");
            for(uint32_t i=0;i<n;++i)for(size_t table=0;table<2;++table)
                tables[table].ids.insert(tables[table].ids.end(),rows[i][table].begin(),rows[i][table].end());
        }
        host_plan=4*1024*1024+mask_bytes; // IO reserve plus caller's borrowed mask
        for(size_t i=0;i<2;++i){
            auto& t=tables[i];std::sort(t.ids.begin(),t.ids.end());
            t.ids.erase(std::unique(t.ids.begin(),t.ids.end()),t.ids.end());
            require(!t.ids.empty()&&uint64_t(t.ids.back())<layout.rows[i],"Prompt Engram ID exceeds original table");
            host_plan=add(host_plan,add(uint64_t(t.ids.capacity())*sizeof(uint32_t),uint64_t(t.ids.size())*row_bytes));
        }
        require(host_plan<=admitted_host_bytes,"Whole-prompt Engram rows exceed explicit host admission");
        bases=offsets;prepared=true;
    }
    uint64_t budget()const{return prepared?host_plan:0;}
    // LLAMADECODE_ENGRAM_EARLY=auto: the early loader's read requests for the
    // prepared rows (its 64 KiB extents, both tables; prior-cache hits ignored).
    uint64_t planned_requests()const{
        require(prepared,"Engram read requests need prepared rows");
        uint64_t n=0;
        for(size_t t=0;t<2;++t){
            std::vector<uint8_t> held;
            if(row_cache){
                std::lock_guard<std::mutex> lock(row_cache->mutex);held.assign((tables[t].ids.size()+7)/8,0);
                for(size_t i=0;i<tables[t].ids.size();++i)if(row_cache->contains(uint32_t(t),tables[t].ids[i]))held[i>>3]|=uint8_t(1u<<(i&7));
            }
            size_t pos=0;llama_dsv41_engram_plan::extent e;
            while(llama_dsv41_engram_plan::next_extent(tables[t].ids.data(),tables[t].ids.size(),held.empty()?nullptr:held.data(),pos,bases[t],row_bytes,
                  large_bytes?large_bytes:llama_dsv41_engram_plan::default_merge_cap,e,engram_read_alignment(),merge_gap))++n;
        }
        return n;
    }
    // One 4 KiB-aligned read per row on a single queue, publishing table 0 when
    // its last row lands. The synchronous arm uses the early loader instead
    // (prepare_early/load_early); this stays as the reference the early rows are
    // byte-compared against.
    void load(const std::filesystem::path& path,int verified_fd,uint64_t verified_file_bytes,const std::function<bool()>& stopped){
        require(prepared&&!load_attempted&&verified_fd>=0,"Engram rows must be prepared/admitted once before load");load_attempted=true;
        require(!stopped||!stopped(),"Engram preload cancelled before IO");
        io_scope io;
        io.fd=open_verified_row_file(path,verified_fd,verified_file_bytes);
        io.begin(2*depth,uint64_t(depth)*page_size);
        for(auto& t:tables){t.data.resize(t.ids.size()*row_bytes);require(t.data.capacity()==t.data.size(),"Unexpected raw-row cache allocation capacity");}
        struct slot {uint32_t table=0,extent=0,skip=0;size_t row=0;uint64_t id=0;bool live=false;};
        std::array<slot,depth> slots;
        size_t next_row=0;uint32_t next_table=0;uint64_t sequence=0;
        uint64_t read_bytes=0,read_requests=0,submits=0;size_t table0_done=0;
        const char*batch_env=std::getenv("LLAMA_DSV41_ENGRAM_SUBMIT_BATCH");
        const bool batch_submits=batch_env&&std::strcmp(batch_env,"1")==0;
        auto issue=[&](uint32_t i){
            while(next_table<2&&next_row==tables[next_table].ids.size()){++next_table;next_row=0;}
            if(next_table==2)return false;
            auto& s=slots[i];require(!s.live,"Engram read queue reused a live slot");
            s.table=next_table;s.row=next_row++;
            const uint64_t offset=add(bases[s.table],uint64_t(tables[s.table].ids[s.row])*row_bytes);
            const uint64_t begin=offset&~uint64_t(4095);s.skip=uint32_t(offset-begin);
            s.extent=(s.skip+row_bytes+4095)&~4095u;
            require(s.extent<=page_size&&begin<=verified_file_bytes&&s.extent<=verified_file_bytes-begin,"Engram aligned read exceeds original file");
            require(sequence<(std::numeric_limits<uint64_t>::max()>>9),"Engram operation nonce overflow");
            s.id=(++sequence<<9)|(i+1);
            io.queue.request(s.id,io.pages+uint64_t(i)*page_size,s.extent,begin);
            s.live=true;read_bytes+=s.extent;++read_requests;return true;
        };
        auto submit=[&]{
            if(!io.queue.in_sq)return;
            // The queue keeps entries the kernel has not seen, so a short flush
            // loses nothing and needs no re-enqueue.
            require(io.queue.flush(),"Required Engram read queue submit failed; no serial fallback");
            ++submits;
        };
        const auto began=engram_clock::now();
        for(uint32_t i=0;i<depth;++i)if(!issue(i))break;
        submit();
        while(io.queue.outstanding()){
            require((!stopped||!stopped())&&engram_ms_since(began)<120000,"Engram preload cancelled or exceeded120s");
            uint32_t completed=0;read_queue::completion c;
            for(;completed<64;++completed){
                if(!io.queue.peek(c))break;
                const auto id=uint32_t(c.id&511u);
                require(id&&id<=depth,"Engram completion has an unknown slot");
                auto& s=slots[id-1];
                require(s.live&&s.id==c.id,"Engram completion nonce/ownership differs");s.live=false;
                require(c.res==int(s.extent),"Engram read failed or was short");
                std::memcpy(tables[s.table].data.data()+s.row*row_bytes,io.pages+uint64_t(id-1)*page_size+s.skip,row_bytes);
                if(s.table==0&&++table0_done==tables[0].ids.size()){
                    ready_tables.store(1,std::memory_order_release);
                    llama_dsv41_layer_log("DS4PERF_ENGRAM_TABLE_LOADED table=0 rows=%zu wall_ms=%.0f\n",tables[0].ids.size(),engram_ms_since(began));
                }
                issue(id-1);
            }
            require(io.queue.outstanding()<=depth,"Engram queued and submitted reads exceed fixed depth");
            // Keep the same bounded slot ring. Amortize tiny completion groups,
            // but replenish before outstanding reads fall below half its depth.
            if(!batch_submits||io.queue.in_sq>=64||io.queue.in_kernel<=depth/2)submit();
            if(!completed&&io.queue.outstanding())io.queue.wait(io.event_fd);
        }
        require(next_table==2&&io.queue.outstanding()==0&&table0_done==tables[0].ids.size(),"Incomplete whole-prompt Engram preload");
        ready_tables.store(2,std::memory_order_release);
        if(batch_submits)llama_dsv41_layer_log("DS4PERF_ENGRAM_SUBMIT_BATCH requests=%llu submits=%llu depth=256 threshold=64 low_water=128 all_completions_drained=1\n",(unsigned long long)read_requests,(unsigned long long)submits);
        llama_dsv41_layer_log("[dsv41-engram-prefetch] table0_rows=%zu table1_rows=%zu host_plan_bytes=%llu read_bytes=%llu requests=%llu submits=%llu wall_ms=%.0f reader=%s\n",
            tables[0].ids.size(),tables[1].ids.size(),(unsigned long long)host_plan,(unsigned long long)read_bytes,
            (unsigned long long)read_requests,(unsigned long long)submits,engram_ms_since(began),io.queue.uring?"io_uring":"pread");
        std::fflush(stderr);
    }
    // ---- LLAMADECODE_ENGRAM_EARLY: same rows/ids/order as prepare(); the IO
    // arena is sized from the admission headroom left by the actual rows. ----
    static uint32_t env_u32(const char*name,uint32_t fallback,uint32_t lo,uint32_t hi){
        const char*e=std::getenv(name);if(!e||!*e)return fallback;
        char*end=nullptr;const unsigned long v=std::strtoul(e,&end,10);
        require(end&&!*end&&v>=lo&&v<=hi,"LLAMADECODE_ENGRAM_* setting out of range");return uint32_t(v);
    }
    static constexpr uint32_t small_slot=8192,large_slot=llama_dsv41_engram_plan::default_merge_cap;
    void prepare_early(const llama_dsv41::engram_layout& layout,const int32_t* tokens,uint32_t count,
                       const std::array<uint64_t,2>& offsets,uint64_t admitted,llama_dsv41::engram_history initial,
                       std::shared_ptr<const llama_dsv41_engram_prompt_cache> prior_cache,const uint8_t* text_mask){
        prepare(layout,tokens,count,offsets,admitted,initial,text_mask); // host_plan holds the fixed 4 MiB IO reserve
        early_mode=true;prior=std::move(prior_cache);
        early_rings=env_u32("LLAMADECODE_ENGRAM_RINGS",4,1,8);
        const uint32_t depth=env_u32("LLAMADECODE_ENGRAM_DEPTH",1024,32,4096);
        const uint64_t rows=uint64_t(tables[0].ids.size())+tables[1].ids.size();
        const uint64_t without=host_plan-4*1024*1024;
        // Slot tables, plan state and (with a prior cache) the hit bitmaps.
        const uint64_t meta=add(256*1024,add(uint64_t(early_rings)*(depth+depth/16+1)*64,prior||row_cache?rows/8+64:0));
        require(add(without,meta)<admitted,"Early Engram metadata exceeds explicit host admission");
        const uint64_t per=(admitted-without-meta)/early_rings;
        // Per ring: n small (8 KiB) slots and max(1,n/16) large (64 KiB) slots.
        require(per>uint64_t(large_slot)+16*(small_slot+large_slot/16),"Early Engram IO arena below its admitted minimum");
        early_small=uint32_t(std::min<uint64_t>(depth,(per-large_slot)/(small_slot+large_slot/16)));
        early_large=std::max(1u,early_small/16);
        early_batch=std::min(env_u32("LLAMADECODE_ENGRAM_BATCH",256,1,4096),std::max(1u,early_small/2));
        early_arena=uint64_t(early_rings)*(uint64_t(early_small)*small_slot+uint64_t(early_large)*large_slot);
        large_bytes=large_slot;merge_gap=0;
        choose_scan(admitted,without,meta);
        host_plan=add(without,add(meta,early_arena));
        require(host_plan<=admitted,"Early Engram rows and IO arena exceed explicit host admission");
    }
    // Extents and bytes for one geometry (cap, gap), both tables, no hits.
    void plan_geometry(uint32_t cap,uint64_t gap,uint64_t& requests,uint64_t& bytes)const{
        requests=bytes=0;
        for(size_t t=0;t<2;++t){
            size_t pos=0;llama_dsv41_engram_plan::extent e;
            while(llama_dsv41_engram_plan::next_extent(tables[t].ids.data(),tables[t].ids.size(),nullptr,pos,bases[t],row_bytes,cap,e,engram_read_alignment(),gap)){++requests;bytes+=e.bytes;}
        }
    }
    // LLAMADECODE_ENGRAM_SCAN: auto (default) reads by large extents when the
    // predicted sequential time (bytes / LLAMADECODE_ENGRAM_SCAN_GBPS, default
    // 15 GB/s, measured 15-22 on D:) beats the random-read prediction
    // (requests / LLAMADECODE_ENGRAM_IOPS). Measured on D: (PAIRFIX iobench):
    // random 264 B rows saturate at 150-285K IOPS, so a 200K prompt's ~3M row
    // reads take 26 s, while both 101 GB tables stream in ~10-13 s.
    void choose_scan(uint64_t admitted,uint64_t without,uint64_t meta){
        const char*e=std::getenv("LLAMADECODE_ENGRAM_SCAN");
        const bool off=e&&!std::strcmp(e,"0"),forced=e&&!std::strcmp(e,"1");
        require(!e||!*e||off||forced||!std::strcmp(e,"auto"),"LLAMADECODE_ENGRAM_SCAN must be 0, 1 or auto");
        if(off)return;
        const uint32_t kib=env_u32("LLAMADECODE_ENGRAM_SCAN_KIB",1024,256,32768);
        require(kib%4==0,"LLAMADECODE_ENGRAM_SCAN_KIB must be a multiple of 4");
        const uint32_t scan_cap=kib*1024;
        // Merge tolerance. Defaults measured at 200K real text (PAIRFIX cand3 vs
        // cand4): 8 MiB straight-through extents read 200.8 GB, prompt 46.57 s;
        // 1 MiB extents with a 64 KiB gap read 119.9 GB, prompt 44.44 s. One ring
        // of 64 slots (cand6, 43.17 s) matches the bench optimum: on this RAID one
        // queue at QD64 beats several queues for 256 KiB+ random reads.
        const uint64_t scan_gap=uint64_t(env_u32("LLAMADECODE_ENGRAM_SCAN_GAP_KIB",64,4,32768))*1024;
        const char*g=std::getenv("LLAMADECODE_ENGRAM_SCAN_GBPS");
        double gbps=15.0;
        if(g&&*g){char*end=nullptr;gbps=std::strtod(g,&end);require(end&&!*end&&gbps>=0.5&&gbps<=100,"LLAMADECODE_ENGRAM_SCAN_GBPS must be 0.5-100");}
        uint64_t random_bytes=0;
        plan_geometry(llama_dsv41_engram_plan::default_merge_cap,0,geometry_random_requests,random_bytes);
        plan_geometry(scan_cap,scan_gap,geometry_scan_requests,geometry_scan_bytes);
        geometry_random_s=double(geometry_random_requests)/llama_dsv41_engram_early_iops();
        // A scan is bound by bytes or by its own request rate, whichever is slower.
        geometry_scan_s=std::max(double(geometry_scan_bytes)/(gbps*1e9),double(geometry_scan_requests)/llama_dsv41_engram_early_iops());
        const bool want=forced||geometry_scan_s<geometry_random_s;
        const uint32_t rings=env_u32("LLAMADECODE_ENGRAM_SCAN_RINGS",1,1,8);
        uint32_t slots=env_u32("LLAMADECODE_ENGRAM_SCAN_DEPTH",64,2,256);
        constexpr uint32_t scan_small=64;
        auto arena=[&](uint32_t n){return uint64_t(rings)*(uint64_t(scan_small)*small_slot+uint64_t(n)*scan_cap);};
        // Fit the admission; never silently: the geometry line reports it.
        while(want&&slots>2&&add(without,add(meta,arena(slots)))>admitted)--slots;
        const bool fits=add(without,add(meta,arena(slots)))<=admitted;
        llama_dsv41_layer_log("LLAMADECODE_ENGRAM_GEOMETRY chosen=%s forced=%d random_requests=%llu random_bytes=%llu random_predicted_s=%.3f scan_requests=%llu scan_bytes=%llu "
            "scan_predicted_s=%.3f scan_kib=%u scan_gbps=%.2f rings=%u slots=%u fits=%d\n",want&&fits?"scan":"random",int(forced),
            (unsigned long long)geometry_random_requests,(unsigned long long)random_bytes,geometry_random_s,(unsigned long long)geometry_scan_requests,
            (unsigned long long)geometry_scan_bytes,geometry_scan_s,kib,gbps,rings,slots,int(fits));
        require(!forced||fits,"LLAMADECODE_ENGRAM_SCAN=1 arena exceeds the explicit Engram host admission");
        if(!want||!fits)return;
        scan=true;large_bytes=scan_cap;merge_gap=scan_gap;early_rings=rings;early_small=scan_small;early_large=slots;
        early_batch=std::max(1u,(scan_small+slots)/4);early_arena=arena(slots);
    }
    bool scan_geometry()const{return scan;}
    double predicted_seconds()const{return scan?geometry_scan_s:geometry_random_s;}
    struct early_slot {uint64_t file_offset=0;uint32_t bytes=0,table=0,first=0,last=0,rows=0;uint64_t id=0;bool live=false;};
    struct early_ring {
        read_queue queue;
        uint8_t* arena=nullptr;
        std::vector<early_slot> slots;std::vector<uint32_t> free_small,free_large;
    };
    struct early_scope {
        int fd=-1,event_fd=-1;
        std::vector<early_ring> rings;
        early_scope(){event_fd=::eventfd(0,EFD_NONBLOCK);require(event_fd>=0,"Cannot create Engram completion event");}
        early_scope(const early_scope&)=delete;early_scope&operator=(const early_scope&)=delete;
        ~early_scope(){
            // Unsubmitted entries are abandoned when the ring closes; every
            // submitted read is settled above before its arena is freed.
            for(auto& r:rings)r.queue.settle(event_fd);
            for(auto& r:rings){r.queue.close();if(r.arena)std::free(r.arena);}
            if(fd>=0)::close(fd);
            if(event_fd>=0)::close(event_fd);
        }
    };
    void load_early(const std::filesystem::path& path,int verified_fd,uint64_t verified_file_bytes,const std::function<bool()>& stopped){
        namespace ep=llama_dsv41_engram_plan;
        require(prepared&&early_mode&&!load_attempted&&verified_fd>=0,"Early Engram rows must be prepared/admitted once before load");load_attempted=true;
        require(!stopped||!stopped(),"Engram preload cancelled before IO");
        const auto began=engram_clock::now();
        auto since=[&](engram_clock::time_point t){return engram_ms_since(t);};
        const bool weight_at_start=llama_dsv41_engram_weight_pass.load(std::memory_order_relaxed);
        for(auto& t:tables){t.data.resize(t.ids.size()*row_bytes);require(t.data.capacity()==t.data.size(),"Unexpected raw-row cache allocation capacity");}
        // Previous-request reuse (one deep): rows the provider's last completed
        // cache already holds are copied, not read. The caller keeps that
        // cache's charge counted until this load ends.
        std::array<std::vector<uint8_t>,2> hits;std::array<uint64_t,2> hit_rows{};
        int reuse=prior?0:-1;
        if(prior){
            const implementation* q=prior->p.get();
            if(q&&q->ready_tables.load(std::memory_order_acquire)==2&&q->bases==bases){
                reuse=1;
                for(size_t t=0;t<2;++t){
                    const auto& mine=tables[t].ids;const auto& old=q->tables[t].ids;hits[t].assign((mine.size()+7)/8,0);
                    for(size_t i=0,j=0;i<mine.size()&&j<old.size();){
                        if(mine[i]<old[j])++i;else if(old[j]<mine[i])++j;
                        else{std::memcpy(tables[t].data.data()+i*row_bytes,q->tables[t].data.data()+j*row_bytes,row_bytes);
                             hits[t][i>>3]|=uint8_t(1u<<(i&7));++hit_rows[t];++i;++j;}
                    }
                }
            }
            prior.reset();
        }
        // LLAMADECODE_ENGRAM_CACHE_MB: rows the process-wide row cache holds are
        // copied (and made most recent); only the rest are read below.
        uint64_t cache_hits=0;double cache_lookup_ms=0;
        if(row_cache){
            const auto looked=engram_clock::now();std::lock_guard<std::mutex> lock(row_cache->mutex);
            for(size_t t=0;t<2;++t){
                const auto& mine=tables[t].ids;if(hits[t].empty())hits[t].assign((mine.size()+7)/8,0);
                for(size_t i=0;i<mine.size();++i){
                    if((hits[t][i>>3]>>(i&7))&1u)continue;
                    if(row_cache->lookup(uint32_t(t),mine[i],tables[t].data.data()+i*row_bytes)){hits[t][i>>3]|=uint8_t(1u<<(i&7));++hit_rows[t];++cache_hits;}
                }
            }
            cache_lookup_ms=since(looked);
        }
        const std::array<uint64_t,2> need{tables[0].ids.size()-hit_rows[0],tables[1].ids.size()-hit_rows[1]};
        std::array<uint64_t,2> done{};
        double table0_ms=-1;
        if(!need[0]){ready_tables.store(1,std::memory_order_release);table0_ms=since(began);}
        early_scope io;
        io.fd=open_verified_row_file(path,verified_fd,verified_file_bytes);
        // LLAMADECODE_ENGRAM_READ_ALIGN=512 asks for the smaller read geometry.
        // Windows checked it against the volume's logical sector because its
        // reads were unbuffered; these are buffered, so any alignment the
        // planner accepts is readable and only the geometry is reported.
        if(engram_read_alignment()!=4096)
            std::fprintf(stderr,"PERFEXP_E13 read_alignment=%u extents=buffered\n",engram_read_alignment());
        const uint32_t slots_per_ring=early_small+early_large;
        const uint64_t ring_arena=uint64_t(early_small)*small_slot+uint64_t(early_large)*large_bytes;
        require(ring_arena<=UINT32_MAX,"Early Engram ring arena exceeds one allocation");
        auto slot_offset=[&](uint32_t s){return s<early_small?uint64_t(s)*small_slot:uint64_t(early_small)*small_slot+uint64_t(s-early_small)*large_bytes;};
        io.rings.resize(early_rings);
        for(auto& r:io.rings){
            r.queue.fd=io.fd;
            r.queue.open(2*slots_per_ring,io.event_fd);
            r.arena=allocate_arena(size_t(ring_arena));
            r.slots.resize(slots_per_ring);
            for(uint32_t s=early_small;s-->0;)r.free_small.push_back(s);
            for(uint32_t s=slots_per_ring;s-->early_small;)r.free_large.push_back(s);
        }
        const double setup_ms=since(began);
        // Plan stream: table 0's extents first, then table 1's, in file order.
        uint32_t cur_table=0;size_t cur_pos=0;bool have_pending=false;ep::extent pending;uint32_t pending_table=0;
        auto fetch=[&]{
            if(have_pending)return true;
            while(cur_table<2){
                const auto& t=tables[cur_table];
                if(ep::next_extent(t.ids.data(),t.ids.size(),hits[cur_table].empty()?nullptr:hits[cur_table].data(),cur_pos,bases[cur_table],row_bytes,large_bytes,pending,engram_read_alignment(),merge_gap)){
                    pending_table=cur_table;have_pending=true;return true;}
                ++cur_table;cur_pos=0;
            }
            return false;
        };
        uint64_t sequence=0;uint64_t read_bytes=0,requests=0,submits=0,useful_rows=0,max_inflight=0,replanned=0;
        auto issue=[&](early_ring& r){
            while(fetch()){
                if(pending.bytes>small_slot&&r.free_large.empty()){
                    // No large slot free: wait for one while this ring still
                    // has a quarter of its slots in flight; otherwise re-plan
                    // the same rows from the same index at the small-slot cap
                    // (same rows land, more requests) instead of starving.
                    if(scan||r.queue.outstanding()>=early_small/4)return; // scan: always wait for a large slot
                    const auto& t=tables[pending_table];cur_pos=pending.first;
                    require(ep::next_extent(t.ids.data(),t.ids.size(),hits[pending_table].empty()?nullptr:hits[pending_table].data(),cur_pos,bases[pending_table],row_bytes,small_slot,pending,engram_read_alignment()),"Engram re-plan lost its rows");
                    ++replanned;
                }
                auto& pool=pending.bytes>small_slot?r.free_large:r.free_small;
                if(pool.empty())return;
                const uint32_t s=pool.back();pool.pop_back();
                auto& sl=r.slots[s];require(!sl.live,"Engram read queue reused a live slot");
                require(pending.file_offset<=verified_file_bytes&&pending.bytes<=verified_file_bytes-pending.file_offset,"Engram aligned read exceeds original file");
                require(sequence<(std::numeric_limits<uint64_t>::max()>>17),"Engram operation nonce overflow");
                sl.file_offset=pending.file_offset;sl.bytes=pending.bytes;sl.table=pending_table;sl.first=pending.first;sl.last=pending.last;sl.rows=pending.rows;
                sl.id=(++sequence<<16)|(s+1);
                r.queue.request(sl.id,r.arena+slot_offset(s),sl.bytes,sl.file_offset);
                sl.live=true;read_bytes+=sl.bytes;++requests;useful_rows+=sl.rows;have_pending=false;
            }
        };
        auto submit=[&](early_ring& r,bool drain){
            if(!r.queue.in_sq)return;
            // Always batched: submit a full batch, or whatever is queued once
            // the ring's in-flight reads fall to half its slots or the plan ends.
            if(!drain&&r.queue.in_sq<early_batch&&r.queue.in_kernel>slots_per_ring/2)return;
            require(r.queue.flush(),"Required Engram read queue submit failed; no serial fallback");
            ++submits;
        };
        std::vector<const read_queue*> waiting;
        for(const auto& r:io.rings)waiting.push_back(&r.queue);
        auto last=engram_clock::now();double overlap_ms=0;bool weight_active=weight_at_start;
        bool uring=false;for(const auto& r:io.rings)uring=uring||r.queue.uring;
        while(true){
            const auto now=engram_clock::now();
            if(weight_active)overlap_ms+=std::chrono::duration<double,std::milli>(now-last).count();
            last=now;weight_active=llama_dsv41_engram_weight_pass.load(std::memory_order_relaxed);
            require((!stopped||!stopped())&&since(began)<120000,"Engram preload cancelled or exceeded120s");
            bool progress=false;
            for(auto& r:io.rings){
                for(uint32_t k=0;k<1024;++k){
                    read_queue::completion c;
                    if(!r.queue.peek(c))break;
                    const uint32_t s1=uint32_t(c.id&0xFFFFu);
                    require(s1&&s1<=slots_per_ring,"Engram completion has an unknown slot");
                    auto& sl=r.slots[s1-1];
                    require(sl.live&&sl.id==c.id,"Engram completion nonce/ownership differs");sl.live=false;
                    require(c.res==int(sl.bytes),"Engram read failed or was short");
                    auto& t=tables[sl.table];const uint8_t* src=r.arena+slot_offset(s1-1);
                    const uint8_t* bits=hits[sl.table].empty()?nullptr:hits[sl.table].data();
                    for(size_t i=sl.first;i<=sl.last;++i){
                        if(ep::is_hit(bits,i))continue;
                        const uint64_t off=bases[sl.table]+uint64_t(t.ids[i])*row_bytes-sl.file_offset;
                        require(off+row_bytes<=sl.bytes,"Engram row outside its extent");
                        std::memcpy(t.data.data()+i*row_bytes,src+off,row_bytes);
                    }
                    done[sl.table]+=sl.rows;
                    (s1-1<early_small?r.free_small:r.free_large).push_back(s1-1);progress=true;
                }
            }
            if(ready_tables.load(std::memory_order_relaxed)==0&&done[0]==need[0]){
                ready_tables.store(1,std::memory_order_release);table0_ms=since(began);
                llama_dsv41_layer_log("DS4PERF_ENGRAM_TABLE_LOADED table=0 rows=%zu wall_ms=%.3f early=1\n",tables[0].ids.size(),table0_ms);
            }
            for(auto& r:io.rings)issue(r);
            const bool plan_done=!fetch();
            uint64_t inflight=0;
            for(auto& r:io.rings){submit(r,plan_done);inflight+=r.queue.in_kernel;}
            max_inflight=std::max(max_inflight,inflight);
            bool idle=plan_done;for(const auto& r:io.rings)idle=idle&&!r.queue.outstanding();
            if(idle)break;
            if(!progress&&inflight)wait_for_completion(io.event_fd,waiting);
        }
        require(done[0]==need[0]&&done[1]==need[1]&&useful_rows==need[0]+need[1],"Incomplete whole-prompt Engram early preload");
        ready_tables.store(2,std::memory_order_release);
        const double wall=since(began);
        const uint64_t rows=tables[0].ids.size()+tables[1].ids.size();
        llama_dsv41_layer_log("LLAMADECODE_ENGRAM_EARLY_LOAD table0_rows=%zu table1_rows=%zu hit_rows=%llu reuse=%d extents=%llu merged_rows=%llu requests=%llu submits=%llu "
            "read_bytes=%llu useful_bytes=%llu rings=%u slots=%u large_slots=%u batch=%u replanned=%llu arena_bytes=%llu host_plan_bytes=%llu max_inflight=%llu "
            "setup_ms=%.3f table0_ms=%.3f wall_ms=%.3f weight_pass_at_start=%d weight_pass_overlap_ms=%.3f overlap_frac=%.3f geometry=%s slot_bytes=%u reader=%s\n",
            tables[0].ids.size(),tables[1].ids.size(),(unsigned long long)(hit_rows[0]+hit_rows[1]),reuse,(unsigned long long)requests,
            (unsigned long long)(useful_rows-requests),(unsigned long long)requests,(unsigned long long)submits,(unsigned long long)read_bytes,
            (unsigned long long)(rows*row_bytes),early_rings,early_small,early_large,early_batch,(unsigned long long)replanned,(unsigned long long)early_arena,(unsigned long long)host_plan,
            (unsigned long long)max_inflight,setup_ms,table0_ms,wall,int(weight_at_start),overlap_ms,wall>0?overlap_ms/wall:0.0,scan?"scan":"random",large_bytes,
            uring?"io_uring":"pread");
        llama_dsv41_layer_log("[dsv41-engram-prefetch] table0_rows=%zu table1_rows=%zu host_plan_bytes=%llu read_bytes=%llu requests=%llu submits=%llu wall_ms=%.0f early=1\n",
            tables[0].ids.size(),tables[1].ids.size(),(unsigned long long)host_plan,(unsigned long long)read_bytes,
            (unsigned long long)requests,(unsigned long long)submits,wall);
        if(row_cache){
            // Tables are published; insert every row (present keys are only touched).
            const auto inserting=engram_clock::now();uint64_t evicted=0,cached=0,capacity=0;
            {   std::lock_guard<std::mutex> lock(row_cache->mutex);const uint64_t before=row_cache->evictions();
                for(size_t t=0;t<2;++t)for(size_t i=0;i<tables[t].ids.size();++i)row_cache->insert(uint32_t(t),tables[t].ids[i],tables[t].data.data()+i*row_bytes);
                evicted=row_cache->evictions()-before;cached=row_cache->rows();capacity=row_cache->capacity_row_count();}
            const uint64_t misses=need[0]+need[1];
            llama_dsv41_layer_log("LLAMADECODE_ENGRAM_ROWCACHE rows=%llu hits=%llu prior_hits=%llu misses=%llu hit_rate=%.4f read_bytes=%llu requests=%llu engram_wall_ms=%.3f "
                "lookup_ms=%.3f insert_ms=%.3f cached_rows=%llu capacity_rows=%llu evicted=%llu cache_bytes=%llu\n",
                (unsigned long long)rows,(unsigned long long)cache_hits,(unsigned long long)(hit_rows[0]+hit_rows[1]-cache_hits),(unsigned long long)misses,
                rows?double(cache_hits)/double(rows):0.0,(unsigned long long)read_bytes,(unsigned long long)requests,wall,cache_lookup_ms,since(inserting),
                (unsigned long long)cached,(unsigned long long)capacity,(unsigned long long)evicted,(unsigned long long)row_cache->charge());
            row_cache.reset();
        }
        std::fflush(stderr);
    }
    bool read(uint32_t table,uint64_t offset,void* destination,size_t bytes)const{
        if(table>=2||ready_tables.load(std::memory_order_acquire)<=table||bytes!=row_bytes||!destination||offset<bases[table]||(offset-bases[table])%row_bytes)return false;
        const auto id=(offset-bases[table])/row_bytes;if(id>UINT32_MAX)return false;
        const auto& t=tables[table];const auto found=std::lower_bound(t.ids.begin(),t.ids.end(),uint32_t(id));
        if(found==t.ids.end()||*found!=id)return false;
        std::memcpy(destination,t.data.data()+size_t(found-t.ids.begin())*row_bytes,row_bytes);return true;
    }
    void wait_table(uint32_t table,const std::function<bool()>& stopped)const{
        require(table<2&&prepared,"Engram table wait requires a prepared cache");
        while(ready_tables.load(std::memory_order_acquire)<=table){
            require(!failed.load(std::memory_order_acquire),"Whole-prompt Engram preload failed before this table was ready");
            require(!stopped||!stopped(),"Engram table wait cancelled");
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }
};
llama_dsv41_engram_prompt_cache::llama_dsv41_engram_prompt_cache():p(new implementation){}
llama_dsv41_engram_prompt_cache::~llama_dsv41_engram_prompt_cache()=default;
void llama_dsv41_engram_prompt_cache::prepare(const llama_dsv41::engram_layout &l,const int32_t*t,uint32_t n,const std::array<uint64_t,2>&o,uint64_t b,llama_dsv41::engram_history initial,const uint8_t* text_mask){
    p->prepare(l,t,n,o,b,initial,text_mask);
}
uint64_t llama_dsv41_engram_prompt_cache::budget()const{return p->budget();}
void llama_dsv41_engram_prompt_cache::set_row_cache(std::shared_ptr<llama_dsv41_engram_row_cache> c){p->row_cache=std::move(c);}
bool llama_dsv41_engram_prompt_cache::scan_geometry()const{return p->scan_geometry();}
double llama_dsv41_engram_prompt_cache::predicted_seconds()const{return p->predicted_seconds();}
uint64_t llama_dsv41_engram_prompt_cache::planned_requests()const{return p->planned_requests();}
void llama_dsv41_engram_prompt_cache::prepare_early(const llama_dsv41::engram_layout &l,const int32_t*t,uint32_t n,const std::array<uint64_t,2>&o,uint64_t b,llama_dsv41::engram_history initial,
        std::shared_ptr<const llama_dsv41_engram_prompt_cache> prior,const uint8_t* text_mask){
    p->prepare_early(l,t,n,o,b,initial,std::move(prior),text_mask);
}
void llama_dsv41_engram_prompt_cache::load_early(const std::filesystem::path& path,int verified_file,uint64_t n,const std::function<bool()>&stop){
    try{p->load_early(path,verified_file,n,stop);}
    catch(...){p->failed.store(true,std::memory_order_release);throw;}
}
void llama_dsv41_engram_prompt_cache::load(const std::filesystem::path& path,int verified_file,uint64_t n,const std::function<bool()>&stop){
    try{p->load(path,verified_file,n,stop);}
    catch(...){p->failed.store(true,std::memory_order_release);throw;}
}
bool llama_dsv41_engram_prompt_cache::read(uint32_t table,uint64_t offset,void*dst,size_t n)const{return p->read(table,offset,dst,n);}
void llama_dsv41_engram_prompt_cache::wait_table(uint32_t table,const std::function<bool()>&stop)const{p->wait_table(table,stop);}
