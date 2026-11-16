#ifndef DAMN_CORE_MEMORY_STRIDED_SPAN_HPP
#define DAMN_CORE_MEMORY_STRIDED_SPAN_HPP

// StridedSpan<T>: n values of T, stride bytes apart, in someone else's memory, without their type:
//
//   one member of an array of structs           stride sizeof(struct)
//   one component of an array of std::array      stride sizeof(std::array)
//   a contiguous column                          stride sizeof(T)
//
// what a live view of it needs (numpy takes strides) and what an HDF5 memory selection needs
// (a hyperslab with a stride), so either can read or write the owner's data in place. The
// stride is a whole number of T (stride_values), as numpy and HDF5 count in values

#include "damn/core/def.hpp"

#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace damn::core
{
template<typename T>
struct StridedSpan
{
    using value_type = T;
    using byte_type  = std::conditional_t<std::is_const_v<T>, std::byte const, std::byte>;

    T* ptr             = nullptr;
    std::size_t n      = 0;
    std::size_t stride = sizeof(T); // bytes

    NO_DISCARD std::size_t size() const { return n; }
    NO_DISCARD T* data() const { return ptr; }
    NO_DISCARD bool contiguous() const { return stride == sizeof(T); }
    NO_DISCARD std::size_t stride_values() const { return stride / sizeof(T); }

    // the same values, read only (as std::span<T> to std::span<T const>)
    operator StridedSpan<T const>() const
        requires(!std::is_const_v<T>)
    {
        return {ptr, n, stride};
    }

    NO_DISCARD T& operator[](std::size_t const i) const
    {
        return *reinterpret_cast<T*>(reinterpret_cast<byte_type*>(ptr) + i * stride);
    }
};

// n values from first, stride bytes apart: a whole number of T
template<typename T>
NO_DISCARD StridedSpan<T> strided_span(T* const first, std::size_t const n,
                                       std::size_t const stride)
{
    if (stride < sizeof(T) or stride % sizeof(T) != 0)
        throw std::runtime_error("damn::core::strided_span: stride not a whole number of values");
    return {first, n, stride};
}

} // namespace damn::core

#endif // DAMN_CORE_MEMORY_STRIDED_SPAN_HPP
