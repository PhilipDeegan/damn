#ifndef DAMN_CORE_MATH_ARRAY_HPP
#define DAMN_CORE_MATH_ARRAY_HPP

#include "damn/core/def.hpp"
#include "damn/core/meta/for_N.hpp"
#include "damn/core/def/damn_config.hpp"

#include <array>
#include <limits>
#include <vector>
#include <cassert>
#include <cstdint>
#include <utility>
#include <type_traits>
#include <cstddef>
#include <format>

namespace damn::core
{
template<typename Type, std::size_t Size> // std::array::fill is only constexpr in C++20 ffs
constexpr void fill(Type value, std::array<Type, Size>& array)
{
    for (std::size_t i = 0; i < Size; i++)
        array[i] = value;
}

template<std::size_t To_Size, std::size_t From_Size, typename Type>
NO_DISCARD constexpr std::array<Type, To_Size> sized_array(std::array<Type, From_Size> const& from)
{
    static_assert(To_Size <= From_Size, "invalid sized_array Size template, too large");

    if constexpr (To_Size == From_Size)
        return from;

    std::array<Type, To_Size> to{};

    for (std::size_t i = 0; i < to.size(); i++)
        to[i] = from[i];

    return to;
}


template<std::size_t To_Size, typename... Args>
NO_DISCARD constexpr auto as_sized_array(Args&&... args)
{
    auto arr = std::array{std::forward<decltype(args)>(args)...};

    return sized_array<To_Size>(arr);
}

template<typename Type, std::size_t size>
NO_DISCARD constexpr std::array<Type, size> ConstArray(Type val = 0)
{
    static_assert(size <= std::numeric_limits<std::uint8_t>::max(),
                  "ConstArray loop counter is std::uint8_t and would overflow for this size");
    std::array<Type, size> arr{};
    for (std::uint8_t i = 0; i < size; i++)
        arr[i] = val;
    return arr;
}

template<typename Type>
NO_DISCARD std::vector<Type> displacementFrom(std::vector<Type> const& input)
{
    std::vector<Type> displs(input.size());
    Type off = 0;
    for (Type i = 0; i < static_cast<Type>(input.size()); i++)
    {
        displs[i] = off;
        off += input[i];
    }
    return displs;
}


template<typename F>
NO_DISCARD auto generate_from(F&& f, std::size_t from, std::size_t to)
{
    assert(from <= to);
    using value_type = std::decay_t<std::invoke_result_t<F&, std::size_t const&>>;
    std::vector<value_type> v;
    std::size_t count = to - from;
    if (count > 0)
        v.reserve(count);
    for (std::size_t i = from; i < to; ++i)
        v.emplace_back(f(i));
    return v;
}

template<typename F>
NO_DISCARD auto generate_from(F&& f, std::size_t count)
{
    return generate_from(std::forward<F>(f), 0, count);
}


template<typename F, typename Container>
NO_DISCARD auto generate_from(F&& f, Container& container)
{
    using T          = typename std::decay_t<Container>::value_type;
    using value_type = std::decay_t<std::invoke_result_t<F, T&>>;
    std::vector<value_type> vec;
    vec.reserve(container.size());
    for (auto& v : container)
        vec.emplace_back(f(v));
    return vec;
}

template<typename F, typename Container>
NO_DISCARD auto generate_from(F&& f, Container const& container)
{
    using T          = std::decay_t<Container>::value_type;
    using value_type = std::decay_t<std::invoke_result_t<F, T const&>>;
    std::vector<value_type> vec;
    vec.reserve(container.size());
    for (auto& v : container)
        vec.emplace_back(f(v));
    return vec;
}


template<std::size_t Idx, typename F>
NO_DISCARD auto constexpr generate_array__(F& f, auto&& arr)
{
    return f(arr[Idx]);
}

template<typename F, std::size_t... Is>
NO_DISCARD auto constexpr generate_array_(F& f, auto&& arr,
                                          std::integer_sequence<std::size_t, Is...>)
{
    return std::array{generate_array__<Is>(f, arr)...};
}

template<typename F, typename Type, std::size_t Size>
NO_DISCARD auto constexpr generate_from(F&& f, std::array<Type, Size> const& arr)
{
    return generate_array_(f, arr, std::make_integer_sequence<std::size_t, Size>{});
}

template<typename F, typename Type, std::size_t Size>
NO_DISCARD auto constexpr generate_from(F&& f, std::array<Type, Size>& arr)
{
    return generate_array_(f, arr, std::make_integer_sequence<std::size_t, Size>{});
}


template<std::size_t size, typename FN>
constexpr auto ConstArrayFrom(FN fn)
{
    std::array<decltype(fn()), size> arr{};
    for (std::uint8_t i = 0; i < size; i++)
        arr[i] = fn();
    return arr;
}


template<typename T0, typename T1, std::size_t... Is>
bool _array_equals(T0 const& a, T1 const& b, std::index_sequence<Is...> const&&) DAMN_ALL
{
    return (... && (a[Is] == b[Is]));
}

template<typename T, std::size_t S> // array == doesn't exist on GPU!
bool array_equals(std::array<T, S> const& a, std::array<T, S> const& b) DAMN_ALL
{
    return _array_equals(a, b, std::make_index_sequence<S>{});
}


template<typename As, typename T, std::size_t... Is>
auto _array_minus(T const& a, T const& b, std::index_sequence<Is...> const&&) DAMN_ALL
{
    std::array<As, sizeof...(Is)> arr;
    ((arr[Is] = a[Is] - b[Is]), ...);
    return arr;
}

template<typename As, typename T, std::size_t S>
auto array_minus(std::array<T, S> const& a, std::array<T, S> const& b) DAMN_ALL
{
    return _array_minus<As>(a, b, std::make_index_sequence<S>{});
}


template<typename Op, template<typename, std::size_t> typename A0,
         template<typename, std::size_t> typename A1, typename T, std::size_t S>
void array_op(A0<T, S>& a, A1<T, S> const& b)
{
    for_N<S>([&](auto i) { Op{a[i]}(b[i]); });
}

template<typename Op, template<typename, std::size_t> typename A0, typename T, std::size_t S>
void array_op(A0<T, S>& a, T const& b)
{
    for_N<S>([&](auto i) { Op{a[i]}(b); });
}

} // namespace damn::core

#endif // DAMN_CORE_MATH_ARRAY_HPP
