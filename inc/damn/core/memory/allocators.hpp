#ifndef DAMN_CORE_MEMORY_ALLOCATORS_HPP
#define DAMN_CORE_MEMORY_ALLOCATORS_HPP

#include "damn/core/def/damn_config.hpp"
#include "damn/core/def/detail/mkn_avx.hpp"
#include "damn/core/def/detail/mkn_gpu.hpp"
#include "mkn/gpu.hpp"
#include "mkn/kul/alloc.hpp"

#include <cstdint>
#include <memory>
#include <new>
#include <utility>

namespace damn
{
auto constexpr default_allocator_mode()
{
    if constexpr (CompileOptions::WithMknGpuHW)
        return AllocatorMode::GPU_UNIFIED;
    return AllocatorMode::CPU;
}


enum class Allocators : std::uint8_t { cpu, device, aligned };

// what a container's memory is. AllocatorMode is the run's choice (CPU / GPU_UNIFIED), the
//  rest is the container's (particles skip construction, AVX columns are aligned): an
//  AllocatorMode converts, so Vector<T, alloc_mode> and Vector<T, {alloc_mode, false}> both work
struct AllocationSpec
{
    Allocators type = Allocators::cpu;
    bool construct  = true; // false: a resize leaves new values unset (a device fills them)

    constexpr AllocationSpec() = default;
    constexpr AllocationSpec(Allocators const type_, bool const construct_ = true)
        : type{type_}
        , construct{construct_}
    {
    }
    constexpr AllocationSpec(AllocatorMode const mode, bool const construct_ = true)
        : AllocationSpec{mode == AllocatorMode::GPU_UNIFIED ? Allocators::device : Allocators::cpu,
                         construct_}
    {
    }

    constexpr bool operator==(AllocationSpec const&) const = default;
};


#if DAMN_HAVE_MKN_AVX
inline constexpr std::int32_t aligned_bytes = mkn::avx::Options::ALIGN();
#else
inline constexpr std::int32_t aligned_bytes = 32;
#endif

namespace detail
{
    // the memory of each allocation type: declared only, a type without one fails to compile.
    //  The only place naming other libraries' allocators
    template<typename T, Allocators type>
    struct AllocatorBase;

    template<typename T>
    struct AllocatorBase<T, Allocators::cpu>
    {
        using type = std::allocator<T>;
    };

    template<typename T>
    struct AllocatorBase<T, Allocators::aligned>
    {
        using type = mkn::kul::AlignedAllocator<T, aligned_bytes>;
    };

#if DAMN_HAVE_MKN_GPU
    template<typename T>
    struct AllocatorBase<T, Allocators::device>
    {
        using type = mkn::gpu::MknGPUAllocator<T, 32>; // managed memory (host memory on mkn.gpu's
                                                       // CPU backend)
    };
#endif
} // namespace detail


// a spec's allocator: the type's memory, construction per spec.construct
template<typename T, AllocationSpec spec>
class Allocator : public detail::AllocatorBase<T, spec.type>::type
{
public:
    using value_type                 = T;
    auto static constexpr allocation = spec;

    template<typename U>
    struct rebind
    {
        using other = Allocator<U, spec>;
    };

    Allocator() = default;
    template<typename U>
    Allocator(Allocator<U, spec> const&) noexcept
    {
    }

    template<typename U, typename... Args>
    void construct(U* const p, Args&&... args)
    {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
    template<typename U>
    void construct(U* const p)
        requires(spec.construct)
    {
        ::new (static_cast<void*>(p)) U();
    }
    template<typename U>
    void construct(U* const) noexcept
        requires(not spec.construct)
    {
    }

    bool operator==(Allocator const&) const { return true; } // stateless
};


namespace detail
{
    // the default spec keeps std::allocator: its vectors are plain std::vector<T> (hooks and
    //  interfaces typed so)
    template<typename T, AllocationSpec spec>
    struct VectorAllocator
    {
        using type = Allocator<T, spec>;
    };

    template<typename T>
    struct VectorAllocator<T, AllocationSpec{}>
    {
        using type = std::allocator<T>;
    };
} // namespace detail

} // namespace damn

#endif /* DAMN_CORE_MEMORY_ALLOCATORS_HPP */
