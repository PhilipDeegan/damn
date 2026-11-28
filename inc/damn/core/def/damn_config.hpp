#ifndef DAMN_CORE_DEF_DAMN_CONFIG_HPP
#define DAMN_CORE_DEF_DAMN_CONFIG_HPP


#include "damn/core/def/detail/mkn_avx.hpp"
#include "damn/core/def/detail/mkn_gpu.hpp"

#if !defined(DAMN_HAVE_GPU)
#if DAMN_HAVE_MKN_GPU_HW
#define DAMN_HAVE_GPU 1
#endif // DAMN_HAVE_MKN_GPU_HW
#endif // !defined(DAMN_HAVE_GPU)


#if !defined(DAMN_HAVE_GPU)
#define DAMN_HAVE_GPU 0
#endif // !defined(DAMN_HAVE_GPU)


#if DAMN_HAVE_GPU

#define DAMN_DEV __device__
#define DAMN_HST __host__
#define DAMN_ALL DAMN_HST DAMN_DEV
#define DAMN_WITH_GPU(...) __VA_ARGS__

#else // !DAMN_HAVE_GPU

#define DAMN_DEV
#define DAMN_HST
#define DAMN_ALL
#define DAMN_WITH_GPU(...)

#endif // DAMN_HAVE_GPU

// 1 in the device compilation pass of DAMN_ALL code, 0 on the host: lets
//  `if constexpr (DAMN_DEVICE_PASS)` pick device only (atomicAdd...) or host only
//  (std::string, throw...) paths without #if at the call site
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define DAMN_DEVICE_PASS 1
#else
#define DAMN_DEVICE_PASS 0
#endif

#include "damn/core/def/detail/thrust.hpp"

namespace damn
{
struct CompileOptions
{
    static constexpr bool WithMknGpu   = DAMN_HAVE_MKN_GPU;
    static constexpr bool WithMknGpuHW = DAMN_HAVE_MKN_GPU_HW;
    static constexpr bool WithMknAVX   = DAMN_HAVE_MKN_AVX;
    static constexpr bool WithThrust   = DAMN_HAVE_THRUST;
    static constexpr bool WithGpu      = WithMknGpu;
};


} // namespace damn


#endif /*DAMN_CORE_DEF_DAMN_CONFIG_HPP*/
