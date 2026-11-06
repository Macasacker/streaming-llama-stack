#pragma once
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <sys/resource.h>
#include <sys/mman.h>
#include <unistd.h>
#include "llama-dsv41-host-memory.h"

// One anonymous RAM tier whose pages are locked as far as the kernel allows.
// mlock() is all-or-nothing per call and is refused past RLIMIT_MEMLOCK - 8 MiB
// soft and hard for an unprivileged process on Linux - so the tier is locked in
// chunks and the tail the allowance cannot cover stays allocated and merely
// pageable. That is the rule Basalt uses for its resident expert tier, and
// residency here is enforced by the host-memory paging guard (the swap-growth
// stop), not by the page lock. Allocation itself has no file-backed or pageable
// fallback: an mmap failure and an admission refusal still throw, and the
// caller supplies the physical budget.
namespace llama_dsv41_locked_ram {
// First chunk to try, halved at every refusal down to one page. Locking a
// multi-gigabyte tier in one call would lock nothing at all when the allowance
// is smaller than it.
constexpr uint64_t lock_chunk=64ull<<20;
struct lock_state {
    std::mutex mutex;
    uint64_t bytes=0;          // allocated tier bytes: the admission accounting
    uint64_t locked_bytes=0;   // how much of that the kernel actually holds
    std::unordered_map<void*,uint64_t> suspended;
    std::unordered_map<void*,uint64_t> locked;   // mapping -> the bytes it holds locked
};
inline lock_state &state(){static lock_state value;return value;}
inline uint64_t page_bytes(){return uint64_t(sysconf(_SC_PAGESIZE));}
// Locks from `memory` up to `bytes`, stopping at the first refusal. A refusal
// halves the chunk and retries the same offset, so a tier far above the
// allowance still locks the part the allowance permits. Chunks stay whole
// pages: mlock() accounts rounded-up pages, and a partial-page chunk would
// report fewer bytes than the kernel holds. Returns the bytes locked and, via
// `reason`, the errno of the first refusal.
inline uint64_t lock_pages(void *memory,uint64_t bytes,int *reason=nullptr){
    const uint64_t page=page_bytes();
    if(!page||!bytes)return 0;
    if(reason)*reason=0;
    uint64_t locked=0,chunk=std::min<uint64_t>(lock_chunk,bytes);
    while(locked<bytes){
        const uint64_t n=std::min(chunk,bytes-locked);
        if(mlock(static_cast<uint8_t*>(memory)+locked,n)==0){locked+=n;continue;}
        if(reason&&!*reason)*reason=errno;
        if(n==page)break;                             // not even one page fits: nothing left to try
        chunk=std::max(page,(n/2/page)*page);         // keep the retry a whole number of pages
    }
    return locked;
}
// Best effort: the mapping is unmapped right after, and pages that were never
// locked need nothing from us.
inline void unlock_pages(void *memory,uint64_t locked){
    uint64_t done=0;
    while(done<locked){
        const uint64_t n=std::min<uint64_t>(locked-done,lock_chunk);
        if(munlock(static_cast<uint8_t*>(memory)+done,n)!=0)break;
        done+=n;
    }
}
inline uint64_t memlock_allowance_bytes(){
    struct rlimit limit{};
    if(getrlimit(RLIMIT_MEMLOCK,&limit)!=0)return 0;
    if(limit.rlim_cur==RLIM_INFINITY)return std::numeric_limits<uint64_t>::max();
    return uint64_t(limit.rlim_cur);
}
// DS4COMBO fast switch: give an allocation's physical pages back while the
// process stays alive (VA stays reserved, pointers stay valid), and later take
// fresh zeroed pages at the same address, locking as many as the allowance
// allows. Contents are not preserved.
inline void suspend_pages(void *memory,uint64_t bytes){
    auto &record=state();std::lock_guard<std::mutex> guard(record.mutex);
    if(!memory||!bytes||bytes>record.bytes||record.suspended.count(memory))throw std::runtime_error("Locked carry RAM suspend geometry differs");
    const auto held=record.locked.find(memory);
    const uint64_t locked=held==record.locked.end()?0:held->second;
    // Unlocked before the discard: a locked VMA that is discarded refaults
    // against the same allowance. The discard is the only step that can fail
    // here, and it fails only on a range this tier does not map.
    unlock_pages(memory,locked);
    if(madvise(memory,bytes,MADV_DONTNEED)!=0)throw std::runtime_error("Locked carry RAM release failed");
    if(held!=record.locked.end()){record.locked.erase(held);record.locked_bytes-=locked;}
    record.suspended.emplace(memory,bytes);record.bytes-=bytes;
    std::fprintf(stderr,"DS4COMBO_CARRY_RAM_SUSPENDED bytes=%llu active_bytes=%llu unlocked_bytes=%llu same_address=1\n",
        (unsigned long long)bytes,(unsigned long long)record.bytes,(unsigned long long)locked);
}
inline void resume_pages(void *memory,uint64_t bytes){
    auto &record=state();std::lock_guard<std::mutex> guard(record.mutex);
    const auto pending=record.suspended.find(memory);
    if(!memory||!bytes||pending==record.suspended.end()||pending->second!=bytes||bytes>UINT64_MAX-record.bytes)
        throw std::runtime_error("Locked carry RAM resume geometry differs");
    // Refuse a resume that exceeds available physical memory before taking
    // pages, as the working-set arm did. Pages refault on first touch, so
    // nothing else on this path refuses an oversized resume.
    const uint64_t available=llama_dsv41_host_memory::proc_field("/proc/meminfo","MemAvailable:",1024);
    if(!available||bytes>available)
        throw std::runtime_error("Resumed carry RAM exceeds available physical memory");
    const uint64_t locked=lock_pages(memory,bytes);
    record.locked.insert_or_assign(memory,locked);record.locked_bytes+=locked;
    record.suspended.erase(pending);record.bytes+=bytes;
    std::fprintf(stderr,"DS4COMBO_CARRY_RAM_RESUMED bytes=%llu active_bytes=%llu locked_bytes=%llu same_address=1\n",
        (unsigned long long)bytes,(unsigned long long)record.bytes,(unsigned long long)locked);
}
inline std::shared_ptr<uint8_t[]> allocate(uint64_t bytes,uint64_t admission) {
    if(!bytes||bytes>admission||bytes>std::numeric_limits<size_t>::max())
        throw std::runtime_error("Carry RAM admission failed before allocation");
    const uint64_t page=page_bytes();
    if(!page||bytes>std::numeric_limits<size_t>::max()-(page-1))
        throw std::runtime_error("Carry RAM alignment overflow");
    const size_t size=size_t((bytes+page-1)/page*page);
    if(size>admission)throw std::runtime_error("Aligned carry RAM exceeds admission");
    auto &record=state();std::unique_lock<std::mutex> guard(record.mutex);
    if(record.bytes>std::numeric_limits<uint64_t>::max()-size)
        throw std::runtime_error("Locked carry byte counter overflow");
    void *memory=mmap(nullptr,size,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    if(memory==MAP_FAILED)throw std::runtime_error("Anonymous carry RAM allocation failed");
    int reason=0;
    const uint64_t locked=lock_pages(memory,size,&reason);
    const uint64_t active_bytes=record.bytes+size;
    std::shared_ptr<uint8_t[]> result;
    try{
        record.bytes+=size;record.locked_bytes+=locked;record.locked.insert_or_assign(memory,locked);
        guard.unlock(); // a failure below invokes the deleter; never hold this mutex then.
        result=std::shared_ptr<uint8_t[]>(static_cast<uint8_t*>(memory),[size,locked](uint8_t *pointer){
            auto &held=state();std::lock_guard<std::mutex> release(held.mutex);
            // Unlock the amount this allocation asked for, which may exceed what
            // is locked now after a suspend/resume cycle; munlock on pages that
            // are not locked is a no-op.
            unlock_pages(pointer,locked);
            munmap(pointer,size);
            const auto entry=held.locked.find(pointer);
            if(entry!=held.locked.end()){held.locked_bytes-=entry->second;held.locked.erase(entry);}
            if(!held.suspended.erase(pointer))held.bytes-=size;
        });
    }catch(...){
        // The accounting node or the shared_ptr control block could not be
        // allocated, so no deleter will ever run: release the mapping and the
        // accounting here, then rethrow.
        if(!guard.owns_lock())guard.lock();
        unlock_pages(memory,locked);munmap(memory,size);
        record.locked.erase(memory);record.locked_bytes-=locked;record.bytes-=size;
        throw;
    }
    std::fprintf(stderr,"CARRY_RAM_LOCKED bytes=%llu active_bytes=%llu anonymous=1 locked_bytes=%llu\n",
        (unsigned long long)size,(unsigned long long)active_bytes,(unsigned long long)locked);
    if(locked<size)std::fprintf(stderr,"CARRY_RAM_LOCK_PARTIAL bytes=%llu locked_bytes=%llu pageable_tail_bytes=%llu rlimit_memlock_bytes=%llu errno=%d\n",
        (unsigned long long)size,(unsigned long long)locked,(unsigned long long)(size-locked),
        (unsigned long long)memlock_allowance_bytes(),reason);
    return result;
}
}
