// CPU-only: what the carry RAM tier really is. On Linux the tier is a private
// anonymous mapping whose pages are locked as far as RLIMIT_MEMLOCK allows, so
// this proves: the VMA is anonymous and covers the tier, the pages inside the
// allowance are locked (per-VMA "Locked: kB" in /proc/self/smaps) and resident
// (mincore), a tier larger than the allowance stays allocated with a tail that
// is neither locked nor faulted in, release unmaps it, and an admission refusal
// or an allocation that cannot build its accounting leaves nothing behind.
// Windows proved the lock page by page with QueryWorkingSetEx and the lock was
// all-or-nothing there, so the partial arm has no Windows twin.
#include "../engines/streaming-llama/prefill_llama.cpp/src/llama-dsv41-locked-ram.h"
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <vector>

static bool fail_next_control_allocation=false;
void *operator new(size_t bytes){if(std::exchange(fail_next_control_allocation,false))throw std::bad_alloc();if(void *p=std::malloc(bytes))return p;throw std::bad_alloc();}
void operator delete(void *p)noexcept{std::free(p);}
void operator delete(void *p,size_t)noexcept{std::free(p);}

static uint64_t page_size(){const long p=sysconf(_SC_PAGESIZE);assert(p>0);return uint64_t(p);}
static uint64_t aligned_bytes(uint64_t bytes){const uint64_t page=page_size();return (bytes+page-1)/page*page;}
static uint64_t allowance(){
    struct rlimit limit{};if(getrlimit(RLIMIT_MEMLOCK,&limit)!=0)return 0;
    return limit.rlim_cur==RLIM_INFINITY?std::numeric_limits<uint64_t>::max():uint64_t(limit.rlim_cur);
}

struct region {
    unsigned long long begin=0,end=0;
    unsigned long long locked_kb=0;
    bool anonymous=false;
    bool found=false;
};
// A smaps header line is "begin-end perms offset dev inode pathname"; anything
// else in the file is a field of the VMA above it. The pathname is absent for
// an anonymous mapping - the line then ends in the separator space.
static bool header(const char *line,unsigned long long &a,unsigned long long &b,char *perms,bool &anonymous){
    unsigned long long off=0,ino=0;unsigned int major=0,minor=0;
    if(std::sscanf(line,"%llx-%llx %15s %llx %x:%x %llu",&a,&b,perms,&off,&major,&minor,&ino)<7)return false;
    const char *rest=line;
    for(int seen=0;seen<5;++seen){
        while(*rest&&*rest!=' '&&*rest!='\n')++rest;
        while(*rest==' ')++rest;
    }
    anonymous=(*rest=='\n'||*rest=='\0');
    return true;
}
// The /proc/self/smaps VMA containing `address`, its Locked: kB, and whether it
// has no pathname - which is what an anonymous mapping looks like there.
static bool vma_holding(void *address,region &out){
    FILE *f=std::fopen("/proc/self/smaps","r");
    assert(f);
    char line[512];
    while(std::fgets(line,sizeof line,f)){
        unsigned long long a=0,b=0;char perms[16]={};bool anonymous=false;
        if(!header(line,a,b,perms,anonymous))continue;
        if(!(a<=(unsigned long long)address&&address<(void*)(uintptr_t)b))continue;
        out.begin=a;out.end=b;out.anonymous=anonymous;out.found=true;
        while(std::fgets(line,sizeof line,f)){
            unsigned long long next=0,after=0;char more[16]={};bool unused=false;
            if(header(line,next,after,more,unused))break;   // the next VMA ends this one
            unsigned long long kb=0;
            if(std::sscanf(line,"Locked: %llu",&kb)==1)out.locked_kb=kb;
        }
        break;
    }
    std::fclose(f);return out.found;
}
static bool mapped(unsigned long long begin,unsigned long long end){
    FILE *f=std::fopen("/proc/self/maps","r");
    assert(f);
    char line[512];bool any=false;
    while(std::fgets(line,sizeof line,f)){
        unsigned long long a=0,b=0;char perms[16]={};bool unused=false;
        if(header(line,a,b,perms,unused)&&a<end&&b>begin&&std::strchr(perms,'w')){any=true;break;}
    }
    std::fclose(f);return any;
}
// What the VMAs over [begin,end) add up to. mlock() splits a mapping at the
// lock boundary, so a partly locked tier is several VMAs: every one of them
// must be anonymous, their union must still be the whole tier, and their
// Locked: kB total is what the tier holds locked.
struct coverage {
    unsigned long long bytes=0;      // union of the VMAs inside the range
    unsigned long long locked=0;     // their Locked: kB, in bytes
    unsigned long long count=0;
    bool anonymous=true;
};
static coverage cover(unsigned long long begin,unsigned long long end){
    FILE *f=std::fopen("/proc/self/smaps","r");
    assert(f);
    coverage out;char line[512];bool inside=false;
    unsigned long long locked=0;
    while(std::fgets(line,sizeof line,f)){
        unsigned long long a=0,b=0;char perms[16]={};bool anonymous=false;
        if(header(line,a,b,perms,anonymous)){
            if(inside){out.locked+=locked;++out.count;locked=0;}
            inside=(a<end&&b>begin);
            if(inside){
                if(!anonymous||!std::strchr(perms,'w'))out.anonymous=false;
                const unsigned long long lo=a>begin?a:begin,hi=b<end?b:end;
                if(hi>lo)out.bytes+=hi-lo;
            }
            continue;
        }
        if(!inside)continue;
        unsigned long long kb=0;
        if(std::sscanf(line,"Locked: %llu",&kb)==1)locked=kb*1024;
    }
    if(inside){out.locked+=locked;++out.count;}
    std::fclose(f);return out;
}
// Anonymous writable mappings of exactly `size` bytes. An allocation whose
// accounting throws must not leave one behind.
static unsigned long count_anonymous(unsigned long long size){
    FILE *f=std::fopen("/proc/self/smaps","r");
    assert(f);
    char line[512];unsigned long long count=0;
    while(std::fgets(line,sizeof line,f)){
        unsigned long long a=0,b=0;char perms[16]={};bool anonymous=false;
        if(header(line,a,b,perms,anonymous)&&anonymous&&std::strchr(perms,'w')&&b-a==size)++count;
    }
    std::fclose(f);return count;
}
static bool resident(void *memory,size_t bytes,size_t offset){
    const uint64_t page=page_size();
    const uintptr_t first=(uintptr_t)memory+offset;
    if(first>=(uintptr_t)memory+bytes)return true;
    unsigned char value=0;
    return mincore(reinterpret_cast<void*>(first/page*page),size_t(page),&value)==0&&(value&1)==1;
}
static bool resident_all(void *memory,size_t bytes){
    const uint64_t page=page_size();
    std::vector<unsigned char> flags(size_t(bytes/page),0);
    return mincore(memory,bytes,flags.data())==0&&
        std::all_of(flags.begin(),flags.end(),[](unsigned char v){return (v&1)==1;});
}
static void claim(uint8_t *memory,size_t bytes){assert(memory[0]==0&&memory[bytes-1]==0);memory[0]=19;memory[bytes-1]=23;}

// A tier that fits inside the allowance: every page locked and resident.
static void check_locked_tier(size_t bytes){
    auto memory=llama_dsv41_locked_ram::allocate(bytes,bytes);
    claim(memory.get(),bytes);                                  // zeroed, then writable through the lock
    const unsigned long long begin=(unsigned long long)memory.get(),size=aligned_bytes(bytes);
    region held;assert(vma_holding(memory.get(),held));
    assert(held.found&&held.anonymous&&held.begin==begin);
    assert(held.end-held.begin==size);
    assert(held.locked_kb*1024==size);                          // the whole tier is locked
    const coverage all=cover(begin,begin+size);
    assert(all.anonymous&&all.bytes==size&&all.locked==size&&all.count==1);
    assert(resident_all(memory.get(),bytes));
    assert(llama_dsv41_locked_ram::state().bytes==size);
    assert(llama_dsv41_locked_ram::state().locked_bytes==size);
    memory.reset();
    assert(!mapped(begin,begin+size));                           // the mapping is gone
    assert(llama_dsv41_locked_ram::state().bytes==0&&llama_dsv41_locked_ram::state().locked_bytes==0);
}

// A tier beyond the allowance: allocated, partly locked, tail pageable and not
// faulted in. This is the case the all-or-nothing lock refused outright.
static void check_partial_tier(size_t bytes){
    const uint64_t limit=allowance();
    assert(uint64_t(bytes)>limit);
    auto memory=llama_dsv41_locked_ram::allocate(bytes,bytes);
    assert(memory&&memory[0]==0);
    const unsigned long long begin=(unsigned long long)memory.get(),size=aligned_bytes(bytes);
    region head;assert(vma_holding(memory.get(),head));
    assert(head.found&&head.anonymous&&head.begin==begin);
    assert(head.locked_kb>0);                                   // the allowance is used, not skipped
    assert(head.locked_kb*1024<=limit&&head.locked_kb*1024<size);
    const coverage all=cover(begin,begin+size);
    assert(all.anonymous&&all.bytes==size&&all.count>=2);       // the lock split the mapping
    assert(all.locked==llama_dsv41_locked_ram::state().locked_bytes);
    assert(all.locked<size);
    assert(llama_dsv41_locked_ram::state().bytes==size);
    assert(resident(memory.get(),bytes,0));                     // the locked prefix is present
    const size_t beyond=size_t(all.locked)+size_t(page_size());
    assert(beyond<size);
    assert(!resident(memory.get(),bytes,beyond));               // the tail is neither locked nor touched
    memory.reset();
    assert(!mapped(begin,begin+size));
    assert(llama_dsv41_locked_ram::state().bytes==0&&llama_dsv41_locked_ram::state().locked_bytes==0);
}

int main(int argc,char **argv){
    const size_t requested=argc>1?size_t(std::stoull(argv[1])):2*1024*1024;
    assert(requested&&requested<=(8ull<<30));
    const uint64_t limit=allowance();
    assert(llama_dsv41_locked_ram::state().bytes==0&&llama_dsv41_locked_ram::state().locked_bytes==0);
    // The tier inside the allowance and the tier beyond it are both proven on
    // every run: whichever size is asked for, the other is derived from it.
    const size_t full=size_t(std::min<uint64_t>(aligned_bytes(requested),limit));
    assert(full>=page_size());
    check_locked_tier(full);
    const uint64_t partial=requested>limit?uint64_t(aligned_bytes(requested)):
        uint64_t(aligned_bytes(full))+(64ull<<20);
    const bool partial_run=partial>limit&&partial<=(8ull<<30);
    if(partial_run)check_partial_tier(size_t(partial));
    bool refused=false;
    try{auto wrong=llama_dsv41_locked_ram::allocate(full,full-1);}catch(const std::exception&){refused=true;}
    assert(refused&&llama_dsv41_locked_ram::state().bytes==0&&llama_dsv41_locked_ram::state().locked_bytes==0);
    if(full<=64*1024*1024){
        // The first allocation inside allocate() is the accounting node, the
        // shared_ptr control block the other; whichever throws, the mapping and
        // the accounting must both be released.
        const unsigned long before=count_anonymous(aligned_bytes(full));
        fail_next_control_allocation=true;bool allocation_failed=false;
        try{auto control=llama_dsv41_locked_ram::allocate(full,full);}catch(const std::bad_alloc&){allocation_failed=true;}
        assert(allocation_failed);
        assert(llama_dsv41_locked_ram::state().bytes==0&&llama_dsv41_locked_ram::state().locked_bytes==0);
        assert(count_anonymous(aligned_bytes(full))==before);
        auto first=llama_dsv41_locked_ram::allocate(full,full*2);
        auto second=llama_dsv41_locked_ram::allocate(full,full*2);
        assert(llama_dsv41_locked_ram::state().bytes==2*aligned_bytes(full));
        first.reset();
        region survivor;assert(vma_holding(second.get(),survivor));
        // Two tiers together can exceed the allowance, so the survivor is locked
        // only as far as what is left of it; the tier's own VMA and the tier
        // accounting must still agree.
        assert(survivor.locked_kb*1024==llama_dsv41_locked_ram::state().locked_bytes);
        if(2*aligned_bytes(full)<=limit)assert(survivor.locked_kb*1024==aligned_bytes(full));
        second.reset();
        assert(llama_dsv41_locked_ram::state().bytes==0&&llama_dsv41_locked_ram::state().locked_bytes==0);
    }
    std::printf("PASS: carry tier is private and anonymous, zeroed, writable and fully page-locked "
                "up to the allowance%s; release unmaps it; admission and accounting-allocation refusal "
                "leave nothing behind (memlock allowance %llu MiB, requested tier %zu MiB, "
                "tier beyond the allowance %s)\n",
        uint64_t(requested)>limit?", and allocated with a pageable tail beyond it":"",
        (unsigned long long)(limit/(1024*1024)),requested/(1024*1024),
        partial_run?"proven":"skipped: the allowance covers it");
    return 0;
}
