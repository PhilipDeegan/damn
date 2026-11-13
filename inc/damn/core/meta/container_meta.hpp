#ifndef DAMN_CORE_META_CONTAINER_META_HPP
#define DAMN_CORE_META_CONTAINER_META_HPP

#include <array>
#include <tuple>
#include <vector>
#include <ranges>
#include <concepts>

namespace damn::core
{
template<typename T>
concept IsStdVector
    = std::same_as<T, std::vector<typename T::value_type, typename T::allocator_type>>;

template<typename T> // any size, tuple_size first so other types are false, not errors
concept IsStdArray = requires {
    std::tuple_size<T>::value;
} and std::same_as<T, std::array<typename T::value_type, std::tuple_size<T>::value>>;

/** can be iterated over: begin() and end(), eg a Particles (a Spannable also has data()) */
template<typename T>
concept Iterable = std::ranges::range<T>;

template<typename T, typename Index = int>
concept Subscriptable = requires(T t, Index i) { t[i]; };

} // namespace damn::core

#endif // DAMN_CORE_META_CONTAINER_META_HPP
