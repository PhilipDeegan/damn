#ifndef DAMN_CORE_VECTOR_HPP
#define DAMN_CORE_VECTOR_HPP


#include "damn/core/def/damn_config.hpp"
#include "mkn/kul/meta.hpp"
#include "damn/core/memory/memory.hpp"
#include "damn/core/def/detail/mkn_avx.hpp"
#include "damn/core/def/detail/mkn_gpu.hpp"

#include "mkn/gpu.hpp"
#include "mkn/kul/alloc.hpp"

#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace damn
{


auto constexpr default_allocator_mode()
{
    if constexpr (CompileOptions::WithMknGpuHW)
        return AllocatorMode::GPU_UNIFIED;
    return AllocatorMode::CPU;
}

template<auto allocator_mode>
bool constexpr allocator_mode_supported()
{
    if constexpr (allocator_mode == AllocatorMode::CPU)
        return true;
    else if constexpr (/*allocator_mode == AllocatorMode::GPU
                       or */
                       allocator_mode == AllocatorMode::GPU_UNIFIED)
        return CompileOptions::WithMknGpuHW;
    return false;
}

template<auto _allocator_mode, typename Actual>
struct Allocator
{
    auto static constexpr mode = _allocator_mode;
    using value_type           = Actual;

    auto& operator()() { return actual; }

    Actual actual{};
};

template<typename Type, auto allocator_mode, std::uint8_t mode = 0>
auto constexpr allocator()
{
    static_assert(std::is_same_v<decltype(allocator_mode), AllocatorMode>);
    static_assert(allocator_mode_supported<allocator_mode>());

    if constexpr (allocator_mode == AllocatorMode::CPU)
    {
        if constexpr (mode == 0)
        {
            return Allocator<allocator_mode, typename std::vector<Type>::allocator_type>{};
        }
        else
        {
            if constexpr (CompileOptions::WithMknAVX)
            {
                DAMN_WITH_MKN_AVX(
                    return Allocator<allocator_mode, mkn::kul::AlignedAllocator<
                                                         Type, mkn::avx::Options::ALIGN()>>{};)
            }
            else if constexpr (CompileOptions::WithMknGpu)
            {
                DAMN_WITH_MKN_GPU(
                    return Allocator<allocator_mode,
                                     mkn::kul::NonConstructingHugePageAllocator<Type>>{};)
            }
            else
            {
                return allocator<Type, allocator_mode, 0>(); // default
            }
        }
    }
    else if constexpr (/*allocator_mode == AllocatorMode::GPU
                      or */
                       allocator_mode == AllocatorMode::GPU_UNIFIED)
    {
        if constexpr (CompileOptions::WithMknGpuHW)
        {
            DAMN_WITH_MKN_GPU({
                if constexpr (mode == 0) // for fields
                    return Allocator<allocator_mode, mkn::gpu::ManagedAllocator<Type>>{};
                else // for particles
                    return Allocator<allocator_mode, mkn::gpu::NoConstructAllocator<Type>>{};
            });
        }
        // else compile error
    }
    throw std::runtime_error("NOOO");
}

template<typename Type, auto allocator_mode, std::uint8_t mode>
using allocator_type_t = decltype(allocator<Type, allocator_mode, mode>());

// true if V is a std::vector using an mkn::gpu allocator, for mkn::gpu::{resize,reserve,as_super}
template<typename V>
bool constexpr is_mkn_gpu_vector_v = false;

#if DAMN_HAVE_MKN_GPU
template<typename T, std::int32_t align>
bool constexpr is_mkn_gpu_vector_v<std::vector<T, mkn::gpu::NoConstructAllocator<T, align>>>
    = true;
template<typename T, std::int32_t align>
bool constexpr is_mkn_gpu_vector_v<std::vector<T, mkn::gpu::ManagedAllocator<T, align>>> = true;
#endif // DAMN_HAVE_MKN_GPU

// ^ bi directionality v

template<typename Type, typename allocator_type, std::uint8_t mode = 0>
auto constexpr get_allocator_mode()
{
    if constexpr (std::is_same_v<allocator_type, typename allocator_type_t<Type, AllocatorMode::CPU,
                                                                           mode>::value_type>)
        return AllocatorMode::CPU;
    else
    {
        return AllocatorMode::GPU_UNIFIED; // do better
    }
}


template<typename Type, auto allocator_mode_ = AllocatorMode::CPU, std::uint8_t mode = 0,
         typename vector_t_
         = std::vector<Type, typename allocator_type_t<Type, allocator_mode_, mode>::value_type>>
struct Vector
{
    auto static constexpr allocator_mode = allocator_mode_;

    using value_type     = Type;
    using Allocator_t    = allocator_type_t<Type, allocator_mode, mode>;
    using allocator_type = typename Allocator_t::value_type;
    using vector_t       = vector_t_;
    static_assert(allocator_mode == get_allocator_mode<Type, allocator_type, mode>());

    Vector() = delete; // not meant to be instantiated.

    auto static make(std::size_t size, bool reserve = false)
    {
        static_assert(mkn::kul::IsStdVector<vector_t>);
        vector_t vec(allocator<Type, allocator_mode, mode>()());
        if (size)
        {
            if (reserve)
                vec.reserve(size);
            else
                vec.resize(size);
        }
        return vec;
    }


    auto static make(std::size_t size, Type const& val)
    {
        auto vec = make(size);
        fill(vec, val);
        return vec;
    }


    template<typename Alloc0>
    void static copy(vector_t& dst, std::vector<Type, Alloc0> const& src)
    {
        auto static constexpr src_mode = get_allocator_mode<Type, Alloc0>();

        if (dst.size() != src.size())
            dst.resize(src.size());

        if constexpr (allocator_mode == src_mode)
            dst = src;
        else
            damn::core::gpu::copy(dst.data(), src.data(), src.size());
    }


    template<typename Alloc0>
    void static move(vector_t& dst, std::vector<Type, Alloc0>& src)
    {
        auto static constexpr src_mode = get_allocator_mode<Type, Alloc0>();

        if constexpr (allocator_mode == AllocatorMode::CPU and src_mode == AllocatorMode::CPU)
            dst = std::move(src);

        else
            copy(dst, src);
    }

    template<typename Type0, typename Alloc0>
    auto static from(std::vector<Type0, Alloc0> const& src)
    {
        static_assert(std::is_same_v<Type, Type0>);

        auto static constexpr src_mode = get_allocator_mode<Type0, Alloc0>();


        if constexpr (src_mode == AllocatorMode::CPU)
        {
            return src;
        }
        else
        {
            // allocations shouldn't happen on GPU, so assume CPU
            auto dst = make(src.size());
            copy(dst, src);
            return dst;
        }
    }

    template<typename Type0, typename Alloc0>
    auto static from(std::vector<Type0, Alloc0>&& src)
    {
        static_assert(std::is_same_v<Type, Type0>);
        auto static constexpr src_mode = get_allocator_mode<Type0, Alloc0>();

        if constexpr (allocator_mode == AllocatorMode::CPU and src_mode == AllocatorMode::CPU)
            return std::move(src);

        else
            return from(src);
    }


    auto static fill(vector_t& vec, Type const& val)
    {
        if constexpr (allocator_mode == AllocatorMode::CPU)
            std::fill(vec.begin(), vec.end(), val);

        else
            damn::core::gpu::fill(vec, val);
    }

    auto static fill(value_type* const data, std::size_t const size, Type const val)
    {
        // todo
        // if constexpr (allocator_mode == AllocatorMode::CPU)
        std::fill(data, data + size, val);

        // else
        //     damn::core::gpu::fill(vec, val);
    }
};


template<typename T, auto alloc_mode_ = AllocatorMode::CPU, std::uint8_t mode = 0>
struct MinimizingVector
{
    using vec_helper = damn::Vector<T, alloc_mode_, mode>;
    using vector_t   = vec_helper::vector_t;


    template<bool copy_old = true>
    auto& get(std::size_t const& s)
    {
        if (s < v.capacity() * percentile)
            ++_c;
        else
            _c = 0;

        if (_c == period)
        {
            vector_t r;
            r.reserve(v.capacity() * realloc_to);
            if constexpr (copy_old)
                r = v;
            v  = std::move(r);
            _c = 0;
        }

        v.resize(s);
        return v;
    }


    auto& zero(std::size_t const& s)
    {
        clear();

        if (s < size())
        {
            if (s < v.capacity() * percentile)
                ++_c;
            else
                _c = 0;

            if (_c == period)
            {
                vector_t r;
                r.reserve(v.capacity() * realloc_to);
                v  = std::move(r);
                _c = 0;
            }
        }
        else
            v.reserve(s);

        return v;
    }


    auto& operator[](std::size_t const i) { return v[i]; }
    auto& operator[](std::size_t const i) const { return v[i]; }
    auto& operator()() { return v; }
    auto& operator()() const { return v; }

    auto& clear()
    {
        if (v.size())
            v.clear();
        return v;
    }
    void push_back(auto&&... args) { v.push_back(args...); }
    auto size() const { return v.size(); }
    auto capacity() const { return v.capacity(); }

    void destroy() { v = std::move(vector_t{}); }

    double const percentile  = .80;
    double const realloc_to  = .85;
    std::size_t const period = 20;
    vector_t v;
    std::uint16_t _c = 0;
};


// capacity of v to at least s, the best way for its allocator: mkn::gpu's (unified memory,
//  copy = false: no copy), others std::vector::reserve
template<typename V>
void reserve(V& v, std::size_t const s, bool const copy = true)
{
    if constexpr (is_mkn_gpu_vector_v<V>)
    {
        DAMN_WITH_MKN_GPU(mkn::gpu::reserve(v, s, copy));
    }
    else
        v.reserve(s);
}

// size of v to s, growing as reserve above. mkn.gpu's vectors grow to exactly s (a managed
//  allocation, ~1 ms, each time an array grows a little: appends, a sort's scratch): past the
//  capacity, half as much again is allocated, the size then set within it (not constructed)
template<typename V>
void resize(V& v, std::size_t const s, bool const copy = true)
{
    if constexpr (is_mkn_gpu_vector_v<V>)
    {
        DAMN_WITH_MKN_GPU({
            if (s > v.capacity())
                mkn::gpu::resize(v, s + s / 2, copy);
            mkn::gpu::resize(v, s, copy);
        });
    }
    else
        v.resize(s);
}


} // namespace damn


#endif /* DAMN_CORE_VECTOR_HPP */
