#ifndef DAMN_CORE_META_TYPE_EQUALITY_META_HPP
#define DAMN_CORE_META_TYPE_EQUALITY_META_HPP

#include <concepts>
#include <type_traits>


namespace damn
{
namespace core
{
    // Basic function
    template<typename T>
    constexpr void allsame(T)
    {
    }

    // Recursive function
    template<typename T, typename T2, typename... Ts>
        requires std::same_as<T, T2>
    constexpr void allsame([[maybe_unused]] T arg, T2 arg2, Ts... args)
    {
        allsame(arg2, args...);
    }


} // namespace core

} // namespace damn

#endif
