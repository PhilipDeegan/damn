#ifndef DAMN_CORE_VECTOR_HPP
#define DAMN_CORE_VECTOR_HPP


#include "damn/core/def/damn_config.hpp"
#include "damn/core/memory/memory.hpp"
#include "damn/core/def/detail/mkn_avx.hpp"
#include "damn/core/def/detail/mkn_gpu.hpp"
#include "mkn/gpu.hpp"
#include "mkn/kul/alloc.hpp"
#include "mkn/kul/lang/cpp/compiler/def.hpp"

#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <new>
#include <span>

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


// how vectors of an allocation type grow, fill and copy: only declared here, a specialization per
//  type defines it, so a type without one fails to link rather than being handled the wrong way
//  (a host copy of device data, a device copy of host memory). host: whether the memory is host
//  only (copies between two host types are std::copy)
template<Allocators type>
struct VectorOps
{
    template<typename V>
    static void reserve(V& v, std::size_t s, bool copy);

    template<typename V>
    static void resize(V& v, std::size_t s, bool copy);

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
        static void resize(V& v, std::size_t const s, bool /*copy*/)
        {
            v.resize(s);
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
// managed memory holding device data: regrows copied by mkn.gpu (on the device, synced), not a
//  host loop that would migrate both buffers. Each managed allocation costs ~1 ms, so past the
//  capacity resize allocates half as much again
template<>
struct VectorOps<Allocators::device>
{
    bool static constexpr host = false;

    template<typename V>
    static void regrow(V& v, std::size_t const cap, bool const copy)
    {
        V grown(v.get_allocator());
        grown.reserve(cap);
        grown.resize(v.size()); // constructs on the host unless the spec skips construction
        if (copy and v.size())
            mkn::gpu::copy(grown.data(), v.data(), v.size());
        v.swap(grown);
    }

    template<typename V>
    static void reserve(V& v, std::size_t const s, bool const copy)
    {
        if (s > v.capacity())
            regrow(v, s, copy);
    }

    template<typename V>
    static void resize(V& v, std::size_t const s, bool const copy)
    {
        if (s > v.capacity())
            regrow(v, s + s / 2, copy);
        v.resize(s); // within capacity
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


// a std::vector of a spec's memory, its growth, fills and copies per the spec's type (VectorOps)
template<typename T, AllocationSpec spec = AllocationSpec{}>
class Vector
{
    using Ops = VectorOps<spec.type>;

public:
    using value_type                 = T;
    using allocator_type             = typename detail::VectorAllocator<T, spec>::type;
    using std_vector                 = std::vector<T, allocator_type>;
    using iterator                   = typename std_vector::iterator;
    using const_iterator             = typename std_vector::const_iterator;
    auto static constexpr allocation = spec;

    Vector() = default;
    explicit Vector(std::size_t const n) { v.resize(n); }
    Vector(std::size_t const n, T const& val)
        : Vector(n)
    {
        fill(val);
    }

    Vector(Vector const& that)
        requires(Ops::host)
        : v(that.v)
    {
    }
    Vector(Vector const& that)
        requires(not Ops::host)
    {
        copy_from(that);
    }
    template<AllocationSpec that_spec>
    explicit Vector(Vector<T, that_spec> const& that)
    {
        copy_from(that);
    }
    Vector(Vector&&) noexcept = default;

    Vector& operator=(Vector const& that)
    {
        if (this != &that)
            copy_from(that);
        return *this;
    }
    Vector& operator=(Vector&&) noexcept = default;

    // values copied from that (any spec): std::copy between host memories, else mkn.gpu's copy
    template<AllocationSpec that_spec>
    void copy_from(Vector<T, that_spec> const& that)
    {
        resize(that.size(), /*copy=*/false);
        copy_values(that);
    }

    void resize(std::size_t const s, bool const copy = true) { Ops::resize(v, s, copy); }
    void reserve(std::size_t const s, bool const copy = true) { Ops::reserve(v, s, copy); }
    void fill(T const& val) { Ops::fill(v, val); }

    // the same memory as a constructing vector (emplace_back / push_back of defaulted values on a
    //  non constructing spec): the allocators are stateless and differ only in construct()
    auto& constructing()
    {
        using Constructing
            = std::vector<T, typename detail::VectorAllocator<T, AllocationSpec{spec.type}>::type>;
        return reinterpret_cast<Constructing&>(v);
    }

    std_vector& vec() { return v; }
    std_vector const& vec() const { return v; }

    auto data() { return v.data(); }
    auto data() const { return v.data(); }
    auto size() const { return v.size(); }
    MKN_KUL_FN_SIG_RET bool empty() const { return v.empty(); }
    auto capacity() const { return v.capacity(); }
    auto& operator[](std::size_t const i) { return v[i]; }
    auto& operator[](std::size_t const i) const { return v[i]; }
    auto& front() { return v.front(); }
    auto& front() const { return v.front(); }
    auto& back() { return v.back(); }
    auto& back() const { return v.back(); }
    auto begin() { return v.begin(); }
    auto begin() const { return v.begin(); }
    auto end() { return v.end(); }
    auto end() const { return v.end(); }
    std::span<T> span() { return {v.data(), v.size()}; }
    std::span<T const> span() const { return {v.data(), v.size()}; }

    void clear() { v.clear(); }
    template<typename... Args>
    auto& emplace_back(Args&&... args)
    {
        return v.emplace_back(std::forward<Args>(args)...);
    }
    void push_back(T const& t) { v.push_back(t); }
    void push_back(T&& t) { v.push_back(std::move(t)); }
    void pop_back() { v.pop_back(); }
    void swap(Vector& that) noexcept { v.swap(that.v); }

protected:
    template<AllocationSpec that_spec>
    void copy_values(Vector<T, that_spec> const& that)
        requires(Ops::host and VectorOps<that_spec.type>::host)
    {
        std::copy(that.begin(), that.end(), v.begin());
    }
    template<AllocationSpec that_spec>
    void copy_values(Vector<T, that_spec> const& that)
        requires(not(Ops::host and VectorOps<that_spec.type>::host))
    {
        if (that.size())
            damn::core::gpu::copy(v.data(), that.data(), that.size());
    }

    std_vector v;
};


// a Vector whose capacity follows its recent use: after `period` uses in a row under percentile
//  of capacity, capacity drops to realloc_to of itself
template<typename T, AllocationSpec spec = AllocationSpec{}>
struct MinimizingVector : Vector<T, spec>
{
    using Super = Vector<T, spec>;
    using Super::capacity;
    using Super::size;
    using Super::v;


    template<bool copy_old = true>
    auto& get(std::size_t const& s)
    {
        fit<copy_old>(s);
        Super::resize(s, copy_old);
        return v;
    }


    // s values, all T{}: the old ones not kept, so a shrink does not copy them
    auto& zero(std::size_t const& s)
    {
        fit<false>(s);
        v.clear();
        v.resize(s);
        return v;
    }

    // s: the size needed this use. Public for users that know s only after the use (ScratchVector)
    template<bool copy_old = true>
    void fit(std::size_t const s)
    {
        if (s < v.capacity() * percentile)
            ++_c;
        else
            _c = 0;
        if (_c != period)
            return;
        Super r;
        r.reserve(v.capacity() * realloc_to, /*copy=*/false);
        if constexpr (copy_old)
            r.copy_from(*this);
        v.swap(r.vec());
        _c = 0;
    }


    auto& operator()() { return v; }
    auto& operator()() const { return v; }

    auto& clear()
    {
        if (v.size())
            v.clear();
        return v;
    }

    void destroy() { v = typename Super::std_vector{}; }

    double const percentile  = .80;
    double const realloc_to  = .85;
    std::size_t const period = 20;
    std::uint16_t _c         = 0;
};


} // namespace damn


#endif /* DAMN_CORE_VECTOR_HPP */
