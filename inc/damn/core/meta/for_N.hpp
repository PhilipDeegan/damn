#ifndef DAMN_CORE_META_FOR_N_HPP
#define DAMN_CORE_META_FOR_N_HPP

#include "damn/core/def.hpp"
#include "damn/core/meta/logic.hpp"
#include "damn/core/meta/type_meta.hpp"
#include "damn/core/def/damn_config.hpp"

#include <array>
#include <tuple>
#include <cstdint>
#include <utility>
#include <type_traits>

namespace damn::core
{
template<std::uint16_t N, typename Fn>
constexpr decltype(auto) for_N_fold(Fn&& fn) DAMN_ALL
{
    /*  // how to use: fn gets every index at once, as a pack
        for_N_fold<3>([&](auto... ics) { return array(idx[ics]...); });
    */

    return [&]<std::uint16_t... Is>(std::integer_sequence<std::uint16_t, Is...>) -> decltype(auto) {
        return fn(std::integral_constant<std::uint16_t, Is>{}...);
    }(std::make_integer_sequence<std::uint16_t, N>{});
}

enum class for_N_R_mode {
    make_tuple = 0,
    make_array,
    forward_tuple,
};

template<std::uint16_t N, auto M = for_N_R_mode::make_tuple, typename Fn>
constexpr auto for_N(Fn&& fn) DAMN_ALL
{
    /*  // how to use
        for_N<2>([](auto ic) {
            constexpr auto i = ic();
            // ...
        });
    */

    static_assert(std::is_same_v<decltype(M), for_N_R_mode>);
    using return_type
        = std::decay_t<std::invoke_result_t<Fn&, std::integral_constant<std::uint16_t, 0>>>;

    return for_N_fold<N>([&](auto... ics) {
        if constexpr (std::is_void_v<return_type>)
            (fn(ics), ...);
        else if constexpr (M == for_N_R_mode::make_tuple)
            return std::make_tuple(fn(ics)...);
        else if constexpr (M == for_N_R_mode::make_array)
            // no CTAD: HIP sees std::array's deduction guide as host only in device code
            return std::array<return_type, sizeof...(ics)>{fn(ics)...};
        else if constexpr (M == for_N_R_mode::forward_tuple)
            return std::forward_as_tuple(fn(ics)...);
        else
            static_assert(dependent_false_v<Fn>, "unknown return mode");
    });
}


template<std::uint16_t N, typename Fn>
constexpr auto for_N_make_array(Fn&& fn)
{
    return for_N<N, for_N_R_mode::make_array>(fn);
}


template<std::uint16_t N, typename Fn>
NO_DISCARD constexpr auto for_N_all(Fn&& fn)
{
    return all(for_N<N, for_N_R_mode::make_array>(fn));
}

template<std::uint16_t N, typename Fn>
NO_DISCARD constexpr auto for_N_any(Fn&& fn)
{
    return any(for_N<N, for_N_R_mode::make_array>(fn));
}

} // namespace damn::core

#endif // DAMN_CORE_META_FOR_N_HPP
