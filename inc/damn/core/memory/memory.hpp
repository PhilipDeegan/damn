#ifndef DAMN_CORE_MEMORY_MEMORY_HPP
#define DAMN_CORE_MEMORY_MEMORY_HPP


#include "damn/core/def/kokkos_tools.hpp"
#include "damn/core/def/damn_config.hpp"
#include "damn/core/def/detail/mkn_gpu.hpp"
#include "damn/core/def/detail/raja.hpp"

#include "mkn/gpu.hpp"

#include <cassert>
#include <algorithm>
#include <stdexcept>



namespace damn::core::gpu
{


template<typename T0, typename T1, typename Size>
void copy(T0* dst, T1* src, Size const size)
{
    if constexpr (CompileOptions::WithRAJA)
    {
        DAMN_WITH_RAJA(damn::core::raja::copy(dst, src, size));
    }
    else if (CompileOptions::WithMknGpu)
    {
        DAMN_WITH_MKN_GPU({
            assert(
                not(mkn::gpu::Pointer{dst}.is_host_ptr() and mkn::gpu::Pointer{src}.is_host_ptr()));
            mkn::gpu::copy(dst, src, size);
        })
    }
    else
        throw std::runtime_error("Vector::copy NO ALTERNATIVE");
}


template<typename Vector, typename Value>
void fill(Vector& dst, Value const val)
{
    if constexpr (CompileOptions::WithRAJA)
    {
        DAMN_WITH_RAJA(damn::core::raja::set(dst.data(), val, dst.size()));
    }
    else if (CompileOptions::WithMknGpu)
    {
        DAMN_WITH_MKN_GPU(mkn::gpu::fill(dst, val));
    }
    else if (CompileOptions::WithKokkos)
    {
        DAMN_WITH_KOKKOS({ //
            auto data = dst.data();
            Kokkos::parallel_for("fill", 5, KOKKOS_LAMBDA(int i) { data[i] = val; });
        })
    }
    else
        throw std::runtime_error("Vector::fill NO ALTERNATIVE");
}



} // namespace damn::core::gpu


namespace damn::core::mem
{

template<auto alloc_mode, typename T0, typename T1, typename Size>
void copy(T0* dst, T1* src, Size const size)
{
    if constexpr (any_in(AllocatorMode::GPU_UNIFIED, alloc_mode))
        damn::core::gpu::copy(dst, src, size);
    else
        std::copy(src, src + size, dst);
}

template<auto alloc_mode0, auto alloc_mode1, typename T0, typename T1, typename Size>
void copy(T0* dst, T1* src, Size const size)
{
    if constexpr (any_in(AllocatorMode::GPU_UNIFIED, alloc_mode0, alloc_mode1))
        damn::core::gpu::copy(dst, src, size);
    else
        std::copy(src, src + size, dst);
}

} // namespace damn::core::mem




#endif /* DAMN_CORE_MEMORY_MEMORY_HPP */
