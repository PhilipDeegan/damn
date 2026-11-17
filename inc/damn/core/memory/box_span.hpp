#ifndef DAMN_CORE_MEMORY_BOX_SPAN_HPP
#define DAMN_CORE_MEMORY_BOX_SPAN_HPP

// BoxSpan<T, dim>: a buffer of T over the points of a box, in C order (the last direction
// fastest), in someone else's memory: a field's values over its ghost box. BoxRegion: the part
// of it on a sub box (e.g. the domain), what a restart writes or reads in place, without a copy
// (an n dimensional memory hyperslab)

#include "damn/core/def.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"

#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace damn::core
{
template<typename T, std::size_t dim>
struct BoxSpan
{
    using value_type = T;

    T* ptr = nullptr;
    Box<int, dim> box{}; // the buffer's points

    NO_DISCARD std::size_t size() const { return box.size(); }
    NO_DISCARD T* data() const { return ptr; }

    NO_DISCARD T& operator()(Point<int, dim> const& p) const
    {
        auto const shape = box.shape();
        std::size_t flat = 0;
        for (std::size_t i = 0; i < dim; ++i)
            flat = flat * static_cast<std::size_t>(shape[i])
                   + static_cast<std::size_t>(p[i] - box.lower[i]);
        return ptr[flat];
    }

    operator BoxSpan<T const, dim>() const
        requires(!std::is_const_v<T>)
    {
        return {ptr, box};
    }
};

// the values of span on select, a box inside span.box, in C order
template<typename T, std::size_t dim>
struct BoxRegion
{
    using value_type = T;

    BoxSpan<T, dim> span;
    Box<int, dim> select;

    NO_DISCARD std::size_t size() const { return select.size(); }

    operator BoxRegion<T const, dim>() const
        requires(!std::is_const_v<T>)
    {
        return {span, select};
    }
};

template<typename T, std::size_t dim>
NO_DISCARD BoxRegion<T, dim> region(BoxSpan<T, dim> const& span, Box<int, dim> const& select)
{
    auto const overlap = span.box * select;
    if (!overlap or overlap->size() != select.size())
        throw std::runtime_error("damn::core::region: the selection is not inside the span");
    return {span, select};
}

} // namespace damn::core

#endif // DAMN_CORE_MEMORY_BOX_SPAN_HPP
