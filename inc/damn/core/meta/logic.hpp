#ifndef DAMN_CORE_META_LOGIC_HPP
#define DAMN_CORE_META_LOGIC_HPP

#include "damn/core/def.hpp"
#include "damn/core/meta/type_meta.hpp"
#include "damn/core/def/damn_config.hpp"

#include <cstdlib>
#include <algorithm>

namespace damn::core
{
NO_DISCARD auto constexpr any(auto... bools)
    requires(all_are<bool>(bools...))
{
    return (bools || ...);
}

NO_DISCARD auto constexpr all(auto... bools)
    requires(all_are<bool>(bools...))
{
    return (bools && ...);
}


// calls operator bool() or copies bool
auto constexpr static to_bool = [](auto const& v) { return bool{v}; };


template<typename Container, typename Fn = decltype(to_bool)>
NO_DISCARD auto constexpr all(Container const& container, Fn fn = to_bool)
{
    return std::all_of(container.begin(), container.end(), fn);
}

template<typename Container, typename Fn = decltype(to_bool)>
NO_DISCARD auto constexpr any(Container const& container, Fn fn = to_bool)
{
    return std::any_of(container.begin(), container.end(), fn);
}

template<typename Container, typename Fn = decltype(to_bool)>
NO_DISCARD auto constexpr none(Container const& container, Fn fn = to_bool)
{
    return std::none_of(container.begin(), container.end(), fn);
}


auto constexpr all_in(auto const a, auto&&... ts) DAMN_ALL
{
    return ((a == ts) && ...);
}

auto constexpr any_in([[maybe_unused]] auto const a, auto&&... ts) DAMN_ALL
{
    return ((a == ts) || ...);
}


void inline abort_if(bool b)
{
    if (b)
        std::abort();
}

void inline abort_if_not(bool b)
{
    if (!b)
        std::abort();
}

} // namespace damn::core

#endif // DAMN_CORE_META_LOGIC_HPP
