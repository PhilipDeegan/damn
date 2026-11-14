#ifndef DAMN_CORE_DEF_DETAIL_MKN_GPU_HPP
#define DAMN_CORE_DEF_DETAIL_MKN_GPU_HPP


#include "mkn/gpu/def.hpp"
// #if !defined(DAMN_HAVE_MKN_GPU) && __has_include("mkn/gpu.hpp")
// #undef DAMN_HAVE_MKN_GPU
// #endif


#if !defined(DAMN_HAVE_MKN_GPU) || DAMN_HAVE_MKN_GPU == 1
#if __has_include("mkn/gpu.hpp")

#define DAMN_HAVE_MKN_GPU 1

#define DAMN_HAVE_MKN_GPU_HW (MKN_GPU_ROCM || MKN_GPU_CUDA)
#endif // __has_include("mkn/gpu.hpp")
#endif // !defined(DAMN_HAVE_MKN_GPU)


#if !defined(DAMN_HAVE_MKN_GPU)
#define DAMN_HAVE_MKN_GPU 0
#endif // DAMN_HAVE_MKN_GPU

#if !defined(DAMN_HAVE_MKN_GPU_HW)
#define DAMN_HAVE_MKN_GPU_HW 0
#endif // DAMN_HAVE_MKN_GPU_HW


#if DAMN_HAVE_MKN_GPU
#define DAMN_WITH_MKN_GPU(...) __VA_ARGS__
#endif // DAMN_HAVE_MKN_GPU

#if !defined(DAMN_WITH_MKN_GPU)
#define DAMN_WITH_MKN_GPU(...)
#endif

#endif /* DAMN_CORE_DEF_DETAIL_MKN_GPU_HPP */
