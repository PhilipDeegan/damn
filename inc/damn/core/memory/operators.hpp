#ifndef DAMN_CORE_MEMORY_OPERATORS_HPP
#define DAMN_CORE_MEMORY_OPERATORS_HPP

#include "damn/core/def/detail/mkn_gpu.hpp"
#include "damn/core/memory/allocators.hpp"

#include "mkn/gpu.hpp"
#include "mkn/kul/lang/cpp/compiler/def.hpp"

#include <algorithm>
#include <cstddef>

namespace damn
{
// how vectors of an allocation type grow, fill and copy: only declared here, a specialization per
//  type defines it, so a type without one fails to link rather than being handled the wrong way
//  (a host copy of device data, a device copy of host memory). resize returns the old buffer when
//  freeing it waits for the device (else an empty vector): resize_columns frees all its columns'
//  at once. host: whether the memory is host only (copies between two host types are std::copy)
template<Allocators type>
struct VectorOps
{
    template<typename V>
    static void reserve(V& v, std::size_t s, bool copy);

    template<typename V>
    MKN_KUL_FN_SIG_RET static V resize(V& v, std::size_t s, bool copy);

    template<typename V, typename T>
    static void fill(V& v, T const& val);
};

namespace detail
{
    // host memory: std::vector's own growth (copy = false is not worth a special case)
    struct HostOps
    {
        bool static constexpr host = true;

        template<typename V>
        static void reserve(V& v, std::size_t const s, bool /*copy*/)
        {
            v.reserve(s);
        }

        template<typename V>
        MKN_KUL_FN_SIG_RET static V resize(V& v, std::size_t const s, bool /*copy*/)
        {
            v.resize(s);
            return V(v.get_allocator());
        }

        template<typename V, typename T>
        static void fill(V& v, T const& val)
        {
            std::fill(v.begin(), v.end(), val);
        }
    };
} // namespace detail

template<>
struct VectorOps<Allocators::cpu> : detail::HostOps
{
};

template<>
struct VectorOps<Allocators::aligned> : detail::HostOps
{
};

#if DAMN_HAVE_MKN_GPU_HW
// managed memory holding device data: regrows copy on the device (queued on the default stream,
//  no pointer queries, no sync), not a host loop that would migrate both buffers; freeing the old
//  buffer waits for the device. Each managed allocation costs ~1 ms, so past the capacity resize
//  allocates half as much again
template<>
struct VectorOps<Allocators::device>
{
    bool static constexpr host = false;

    template<typename V>
    MKN_KUL_FN_SIG_RET static V regrow(V& v, std::size_t const cap, bool const copy)
    {
        V grown(v.get_allocator());
        grown.reserve(cap);
        grown.resize(v.size()); // constructs on the host unless the spec skips construction
        if (copy and v.size())
            mkn::gpu::copy_on_device_async(grown.data(), v.data(), v.size());
        v.swap(grown);
        return grown;
    }

    template<typename V>
    static void reserve(V& v, std::size_t const s, bool const copy)
        requires(std::is_trivially_copyable_v<typename V::value_type>)
    {
        if (s > v.capacity())
            (void)regrow(v, s, copy); // the old buffer freed here
    }

    template<typename V>
    MKN_KUL_FN_SIG_RET static V resize(V& v, std::size_t const s, bool const copy)
        requires(std::is_trivially_copyable_v<typename V::value_type>)
    {
        V old(v.get_allocator());
        if (s > v.capacity())
            old = regrow(v, s + s / 2, copy);
        v.resize(s); // within capacity
        return old;
    }

    // objects in managed memory (tiles): no device copy of them, std::vector's growth (moves on
    //  the host)
    template<typename V>
    static void reserve(V& v, std::size_t const s, bool /*copy*/)
        requires(not std::is_trivially_copyable_v<typename V::value_type>)
    {
        v.reserve(s);
    }

    template<typename V>
    MKN_KUL_FN_SIG_RET static V resize(V& v, std::size_t const s, bool /*copy*/)
        requires(not std::is_trivially_copyable_v<typename V::value_type>)
    {
        v.resize(s);
        return V(v.get_allocator());
    }

    template<typename V, typename T>
    static void fill(V& v, T const& val)
    {
        mkn::gpu::fill(v, val);
    }
};
#elif DAMN_HAVE_MKN_GPU
// mkn.gpu's CPU backend allocates host memory
template<>
struct VectorOps<Allocators::device> : detail::HostOps
{
};
#endif

} // namespace damn


namespace damn::core::mem
{
// size values from src to dst, memories as their specs (an AllocatorMode converts): std::copy when
//  both are host only, else mkn.gpu's copy (it finds the direction from the pointers)
template<AllocationSpec dst_spec, AllocationSpec src_spec = dst_spec, typename T0, typename T1,
         typename Size>
void copy(T0* const dst, T1* const src, Size const size)
    requires(VectorOps<dst_spec.type>::host and VectorOps<src_spec.type>::host)
{
    std::copy(src, src + size, dst);
}

template<AllocationSpec dst_spec, AllocationSpec src_spec = dst_spec, typename T0, typename T1,
         typename Size>
void copy(T0* const dst, T1* const src, Size const size)
    requires(not(VectorOps<dst_spec.type>::host and VectorOps<src_spec.type>::host))
{
    DAMN_WITH_MKN_GPU(mkn::gpu::copy(dst, src, size));
}
} // namespace damn::core::mem

#endif /* DAMN_CORE_MEMORY_OPERATORS_HPP */
