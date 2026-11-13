#ifndef DAMN_CORE_META_TYPE_META_HPP
#define DAMN_CORE_META_TYPE_META_HPP

#include "damn/core/def.hpp"
#include "damn/core/def/damn_config.hpp"

#include <tuple>
#include <cstddef>
#include <utility>
#include <type_traits>

namespace damn::core
{
template<typename T>
auto constexpr static dependent_false_v = false;


template<typename T, std::size_t... Is>
constexpr auto gft_helper(std::index_sequence<Is...> const&&)
    -> decltype(std::make_tuple((Is, std::declval<T>())...));

template<typename T, std::size_t N>
constexpr auto get_fixed_tuple() -> decltype(gft_helper<T>(std::make_index_sequence<N>{}));

template<typename T, std::size_t N>
using tuple_fixed_type = decltype(get_fixed_tuple<T, N>());


template<typename T>
auto constexpr all_are(auto&&... ts)
{
    return ((std::is_same_v<T, std::decay_t<decltype(ts)>>) && ...);
}

template<typename T>
auto constexpr any_are(auto&&... ts) DAMN_ALL
{
    return ((std::is_same_v<T, std::decay_t<decltype(ts)>>) || ...);
}

template<typename... Ts>
auto constexpr is_any_of(auto const& t) DAMN_ALL
{
    return ((std::is_same_v<Ts, std::decay_t<decltype(t)>>) || ...);
}

template<typename T, typename... Ts>
auto constexpr is_any_of() DAMN_ALL
{
    return ((std::is_same_v<Ts, T>) || ...);
}


template<typename T> // this is so we can use struct {} initialization with
                     // shared_ptrs/forwarding
struct aggregate_adapter : public T
{
    template<class... Args>
    aggregate_adapter(Args&&... args)
        : T{std::forward<Args>(args)...}
    {
    }
};


template<typename Type>
auto& deref(Type&& type) DAMN_ALL
{
    if constexpr (std::is_pointer_v<std::decay_t<Type>>)
        return *type;
    else
        return type;
}


template<typename T>
struct DekayT
{
    T t;
    using value_type = std::decay_t<decltype(t)>;
};

template<typename T>
DekayT(T t) -> DekayT<T>;

} // namespace damn::core

#endif // DAMN_CORE_META_TYPE_META_HPP
