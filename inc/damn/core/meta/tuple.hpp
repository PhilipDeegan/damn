#ifndef DAMN_CORE_META_TUPLE_HPP
#define DAMN_CORE_META_TUPLE_HPP

#include "damn/core/def.hpp"
#include "damn/core/meta/for_N.hpp"

#include <array>
#include <tuple>
#include <string>
#include <utility>

namespace damn::core
{
template<typename Tuple, typename Func>
void apply(Tuple tuple, Func func) DAMN_ALL
{
    std::apply([&](auto&... args) { (func(args), ...); }, tuple);
}


template<typename T, std::size_t S>
auto static constexpr as_tuple(std::array<T, S> const& arr)
{
    return for_N<S, for_N_R_mode::forward_tuple>([&](auto i) -> auto& { return arr[i]; });
};


template<typename Tuple, std::size_t... Is>
constexpr auto named_pair_seq_(Tuple, std::index_sequence<Is...> const&&)
    -> decltype(std::make_tuple(
        (Is, std::declval<std::pair<std::string, std::tuple_element_t<Is, Tuple>>>())...));

template<typename... Args>
auto constexpr named_pair_seq()
    -> decltype(named_pair_seq_(std::tuple<Args...>{},
                                std::make_index_sequence<sizeof...(Args)>{}));

template<typename... Args>
using NamedTuple = decltype(named_pair_seq<Args...>());

template<typename... Pairs>
auto make_named_tuple(Pairs&&... pairs)
{
    return std::make_tuple(pairs...);
}


template<typename Tuple, std::size_t... Is>
constexpr auto _tuple_element_value_type_refs_(Tuple& tup, std::size_t index,
                                               std::index_sequence<Is...> const&&)
{
    return std::forward_as_tuple(std::get<Is>(tup)[index]...);
}

template<typename Tuple>
auto constexpr _tuple_element_value_type_refs(Tuple& tuple, std::size_t index)
{
    return _tuple_element_value_type_refs_(tuple, index,
                                           std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

template<typename... Args>
struct Zipit
{
    auto operator*() { return _tuple_element_value_type_refs(args, idx); }
    bool operator==(Zipit const& that) const { return idx == that.idx; }
    bool operator!=(Zipit const& that) const { return !(*(this) == that); }
    auto& operator++()
    {
        ++idx;
        return *this;
    }

    std::size_t idx = 0;
    std::tuple<Args...>& args;
};

template<typename... Args>
struct Zipper
{
    using Iter = Zipit<Args...>;

    auto begin() { return Iter{0, tup}; }
    auto end() { return Iter{std::get<0>(tup).size(), tup}; }

    std::tuple<Args...> tup;
};

template<typename... Args>
auto zip(Args&&... args)
{
    return Zipper<Args...>{std::forward_as_tuple(args...)};
}

} // namespace damn::core

#endif // DAMN_CORE_META_TUPLE_HPP
