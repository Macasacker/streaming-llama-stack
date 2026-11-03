#pragma once
// The producer's private session transport: a duplicated descriptor for the
// write end of the parent's pipe, kept out of the CRT's stdout once
// session_output_redirect binds. It carries the one operation both the abort
// reporter and the session pipe need: write every byte or fail.
#include <sys/stat.h>
#include <unistd.h>
#include <cstddef>
#include <cstdint>

namespace producer_wire {

using handle = int;
inline handle error_end(){return STDERR_FILENO;}
inline handle standard_input(){return STDIN_FILENO;}
inline handle standard_output(){return STDOUT_FILENO;}
inline handle unarmed(){return -1;}
inline void release(handle wire){close(wire);}
inline bool is_pipe(handle wire){struct stat st{};return fstat(wire,&st)==0&&S_ISFIFO(st.st_mode);}

// Writes the whole buffer. A short or failed write returns false; the caller
// decides whether a lost diagnostic line is an error (the session pipe) or is
// already lost (the crash reporter, which must not allocate or fail here).
inline bool write_all(handle wire,const void*data,size_t bytes){
    const unsigned char*p=static_cast<const unsigned char*>(data);
    size_t left=bytes;
    while(left){
        const ssize_t sent=::write(wire,p,left);
        if(sent<=0)return false;
        p+=size_t(sent);left-=size_t(sent);
    }
    return true;
}

} // namespace producer_wire
