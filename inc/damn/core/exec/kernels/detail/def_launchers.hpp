// IWYU pragma: private, include "damn/core/exec/kernels/launchers.hpp"

#ifndef DAMN_CORE_EXEC_KERNELS_DETAIL_DEF_LAUNCHERS_HPP
#define DAMN_CORE_EXEC_KERNELS_DETAIL_DEF_LAUNCHERS_HPP

#include "damn/core/def.hpp"
#include "damn/core/def/damn_config.hpp"
#include "damn/core/def/detail/mkn_gpu.hpp"
#include "mkn/gpu.hpp"
#include "mkn/gpu/def.hpp"

#if DAMN_HAVE_MKN_GPU


namespace damn::kernel
{

auto inline warp_size() DAMN_ALL // make runtime interrogation?
{
#if !defined(MKN_GPU_CUDA) || !defined(MKN_GPU_ROCM)
    throw std::runtime_error("warpsize is not available for cpu");
#elif defined(MKN_GPU_ROCM)
    return 32ull; // not always
#elif defined(MKN_GPU_CUDA)
    return 32ull;
#endif
}

auto inline thread_idx() DAMN_DEV
{
    if (CompileOptions::WithGpu)
    {
        return DAMN_WITH_GPU(threadIdx.x);
    }
    // else compile error
}

auto inline block_idx() DAMN_DEV
{
    if (CompileOptions::WithGpu)
    {
        return DAMN_WITH_GPU(blockIdx.x);
    }
    // else compile error
}

auto inline idx() DAMN_DEV // global
{
    if (CompileOptions::WithMknGpu)
    {
        return DAMN_WITH_MKN_GPU(mkn::gpu::idx());
    }
    // else compile error
}

} // namespace damn::kernel

#else

namespace damn::kernel
{
// no gpu: never called, typed so callers' non-dependent expressions compile
inline std::size_t warp_size()
{
    DAMN_ASSERT(false);
    return 0;
}

inline std::uint32_t thread_idx()
{
    DAMN_ASSERT(false);
    return 0;
}

inline std::uint32_t block_idx()
{
    DAMN_ASSERT(false);
    return 0;
}

inline std::size_t idx()
{
    DAMN_ASSERT(false);
    return 0;
}
} // namespace damn::kernel


#endif // DAMN_HAVE_MKN_GPU

#endif /* DAMN_CORE_EXEC_KERNELS_DETAIL_DEF_LAUNCHERS_HPP */
