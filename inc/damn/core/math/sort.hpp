#ifndef DAMN_CORE_MATH_SORT_HPP
#define DAMN_CORE_MATH_SORT_HPP

#include "damn/core/def/damn_config.hpp"

#include <algorithm>

namespace damn::core
{

// sequential in place sort of [from, to), callable from host and device code, for short ranges
//  as the per cell / per tile gap lists of the particle arrays:
//  device: thrust::seq on CUDA, else an insertion sort (std::sort is host only, rocThrust's
//   thrust::seq sort calls memmove, host only too)
//  host: thrust::seq with thrust, else std::sort
#if defined(__HIP_DEVICE_COMPILE__) || !DAMN_HAVE_THRUST
bool constexpr seq_sort_device_insertion = true;
#else
bool constexpr seq_sort_device_insertion = false;
#endif

void seq_sort(auto from, auto to) DAMN_ALL
{
    if constexpr (DAMN_DEVICE_PASS and seq_sort_device_insertion)
    {
        for (auto i = from; i != to; ++i)
            for (auto j = i; j != from and *(j - 1) > *j; --j)
            {
                auto const tmp = *j;
                *j             = *(j - 1);
                *(j - 1)       = tmp;
            }
    }
    else
    {
#if DAMN_HAVE_THRUST
        thrust::sort(thrust::seq, from, to);
#else
        std::sort(from, to);
#endif
    }
}

} // namespace damn::core

#endif /* DAMN_CORE_MATH_SORT_HPP */
