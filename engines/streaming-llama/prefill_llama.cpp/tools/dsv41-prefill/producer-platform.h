#pragma once
// The one place the producer turns a path or an argument into UTF-8 text. On
// Linux both are already native UTF-8 bytes, so this is a pass-through with one
// name; the receipts and the session protocol always speak UTF-8.
#include <filesystem>
#include <stdexcept>
#include <string>

namespace producer_text {

inline std::string argument(const char*text){return std::string(text);}
inline std::string path_text(const std::filesystem::path&p){return p.string();}
// The native narrow encoding is UTF-8 on this platform, so a path needs no
// conversion (and std::filesystem::u8path is deprecated here).
inline std::filesystem::path native_path(const std::string&utf8){return std::filesystem::path(utf8);}

} // namespace producer_text
