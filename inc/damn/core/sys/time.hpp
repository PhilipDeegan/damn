#ifndef DAMN_CORE_SYS_TIME_HPP
#define DAMN_CORE_SYS_TIME_HPP

#include <chrono>
#include <cstdint>

namespace damn::core
{
std::uint64_t inline now_in_microseconds()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

} // namespace damn::core

#endif // DAMN_CORE_SYS_TIME_HPP
