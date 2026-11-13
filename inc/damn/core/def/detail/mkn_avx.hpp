#ifndef DAMN_CORE_DEF_DETAIL_MKN_AVX_HPP
#define DAMN_CORE_DEF_DETAIL_MKN_AVX_HPP

#if !defined(DAMN_HAVE_MKN_AVX)
#if __has_include("mkn/avx.hpp")
#define DAMN_HAVE_MKN_AVX 1

#else // __has_include("mkn/avx.hpp")
#define DAMN_HAVE_MKN_AVX 0

#endif // __has_include("mkn/avx.hpp")
#endif // !defined(DAMN_HAVE_MKN_AVX)

#if DAMN_HAVE_MKN_AVX

#include "mkn/avx.hpp"
#define DAMN_WITH_MKN_AVX(...) __VA_ARGS__

#else

#define DAMN_WITH_MKN_AVX(...)

#endif // DAMN_HAVE_MKN_AVX

#endif /* DAMN_CORE_DEF_DETAIL_MKN_AVX_HPP */
