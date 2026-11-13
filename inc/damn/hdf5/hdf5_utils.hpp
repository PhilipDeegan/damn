#ifndef DAMN_HDF5_HDF5_UTILS_HPP
#define DAMN_HDF5_HDF5_UTILS_HPP

#include "damn/core/meta/container_meta.hpp"

#include <array>
#include <cstddef>
#include <type_traits>

namespace damn::hdf5
{
// a std::array of dimension or 3 elements: written as a dataset of arrays
template<typename T, std::size_t dimension>
inline constexpr bool is_array_dataset = [] {
    if constexpr (core::IsStdArray<T>)
        return std::tuple_size_v<T> == dimension or std::tuple_size_v<T> == 3;
    else
        return false;
}();

} // namespace damn::hdf5

#endif // DAMN_HDF5_HDF5_UTILS_HPP
