#pragma once
// The producer's private session transport: the write end of the parent's pipe,
// kept out of the CRT's stdout after session_output_redirect binds. A Win32
// handle on Windows, a file descriptor here, with the one operation both the
// abort reporter and the session pipe need: write every byte or fail.
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <cstddef>
#include <cstdint>

namespace producer_wire {

#if defined(_WIN32)
using handle = HANDLE;
inline handle error_end(){return GetStdHandle(STD_ERROR_HANDLE);}
inline handle standard_input(){return GetStdHandle(STD_INPUT_HANDLE);}
inline handle standard_output(){return GetStdHandle(STD_OUTPUT_HANDLE);}
inline handle unarmed(){return nullptr;}
inline void release(handle wire){CloseHandle(wire);}
inline bool is_pipe(handle wire){return GetFileType(wire)==FILE_TYPE_PIPE;}
#else
using handle = int;
inline handle error_end(){return STDERR_FILENO;}
inline handle standard_input(){return STDIN_FILENO;}
inline handle standard_output(){return STDOUT_FILENO;}
inline handle unarmed(){return -1;}
inline void release(handle wire){close(wire);}
inline bool is_pipe(handle wire){struct stat st{};return fstat(wire,&st)==0&&S_ISFIFO(st.st_mode);}
#endif

// Writes the whole buffer. A short or failed write returns false; the caller
// decides whether a lost diagnostic line is an error (the session pipe) or is
// already lost (the crash reporter, which must not allocate or fail here).
inline bool write_all(handle wire,const void*data,size_t bytes){
    const unsigned char*p=static_cast<const unsigned char*>(data);
    size_t left=bytes;
    while(left){
#if defined(_WIN32)
        DWORD sent=0;
        if(!WriteFile(wire,p,DWORD(left>UINT32_MAX?UINT32_MAX:left),&sent,nullptr)||!sent)return false;
#else
        const ssize_t sent=::write(wire,p,left);
        if(sent<=0)return false;
#endif
        p+=size_t(sent);left-=size_t(sent);
    }
    return true;
}

} // namespace producer_wire
