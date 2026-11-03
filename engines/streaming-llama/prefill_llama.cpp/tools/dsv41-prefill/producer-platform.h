#pragma once
// The one place the producer converts between the platform's path/argument
// representation and UTF-8 text. Windows hands it wide strings and expects
// UTF-8 for the receipts; Linux hands bytes that are already UTF-8.
#include <filesystem>
#include <stdexcept>
#include <string>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace producer_text {

#if defined(_WIN32)
inline std::string argument(const std::wstring&s){
    const int n=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,s.data(),int(s.size()),nullptr,0,nullptr,nullptr);
    if(n<=0)throw std::runtime_error("Invalid Unicode command argument");
    std::string out(n,'\0');
    if(WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,s.data(),int(s.size()),out.data(),n,nullptr,nullptr)!=n)
        throw std::runtime_error("Cannot encode command argument");
    return out;
}
inline std::string path_text(const std::filesystem::path&p){return argument(p.wstring());}
inline std::filesystem::path native_path(const std::string&utf8){return std::filesystem::u8path(utf8);}
#else
inline std::string argument(const char*text){return std::string(text);}
inline std::string path_text(const std::filesystem::path&p){return p.string();}
// The native narrow encoding is UTF-8 on this platform, so a path needs no
// conversion (and std::filesystem::u8path is deprecated here).
inline std::filesystem::path native_path(const std::string&utf8){return std::filesystem::path(utf8);}
#endif

} // namespace producer_text
