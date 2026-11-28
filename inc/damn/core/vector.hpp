#ifndef DAMN_CORE_VECTOR_HPP
#define DAMN_CORE_VECTOR_HPP


#include "damn/core/memory/allocators.hpp"
#include "damn/core/memory/operators.hpp"

#include "mkn/kul/lang/cpp/compiler/def.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace damn
{


// a std::vector of a spec's memory, its growth, fills and copies per the spec's type (VectorOps)
template<typename T, AllocationSpec spec = AllocationSpec{}>
class Vector
{
    using Ops = VectorOps<spec.type>;

public:
    using value_type                 = T;
    using allocator_type             = detail::VectorAllocator<T, spec>::type;
    using std_vector                 = std::vector<T, allocator_type>;
    using iterator                   = std_vector::iterator;
    using const_iterator             = std_vector::const_iterator;
    auto static constexpr allocation = spec;

    Vector() = default;
    explicit Vector(std::size_t const n) { v.resize(n); }
    Vector(std::size_t const n, T const& val)
        : Vector(n)
    {
        fill(val);
    }

    Vector(Vector const& that) { copy_from(that); }
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

    // that's values (any spec) and at least its capacity: a copy grows as the original would
    //  have, without a regrow (std::vector's copy allocates only the size)
    template<AllocationSpec that_spec>
    void copy_from(Vector<T, that_spec> const& that)
    {
        reserve(that.capacity(), /*copy=*/false);
        copy_values_from(that);
    }

    // that's values only, the capacity as it is (or as they need): between host memories the
    //  elements copy constructed (as std::vector's copy: types without a default constructor or
    //  copy assignment), else a device copy of trivially copyable values
    template<AllocationSpec that_spec>
    void copy_values_from(Vector<T, that_spec> const& that)
        requires(Ops::host and VectorOps<that_spec.type>::host and std::is_copy_assignable_v<T>)
    {
        v.assign(that.begin(), that.end());
    }
    template<AllocationSpec that_spec>
    void copy_values_from(Vector<T, that_spec> const& that)
        requires(Ops::host and VectorOps<that_spec.type>::host and not std::is_copy_assignable_v<T>)
    {
        v.clear(); // assign would need T's copy assignment
        for (auto const& e : that)
            v.emplace_back(e);
    }
    template<AllocationSpec that_spec>
    void copy_values_from(Vector<T, that_spec> const& that)
        requires(not(Ops::host and VectorOps<that_spec.type>::host))
    {
        resize(that.size(), /*copy=*/false);
        if (that.size())
            core::mem::copy<spec, that_spec>(v.data(), that.data(), that.size());
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

    auto cbegin() const { return v.cbegin(); }
    auto cend() const { return v.cend(); }
    auto erase(const_iterator const pos) { return v.erase(pos); }
    auto erase(const_iterator const first, const_iterator const last)
    {
        return v.erase(first, last);
    }
    template<typename... Args>
    auto insert(const_iterator const pos, Args&&... args)
    {
        return v.insert(pos, std::forward<Args>(args)...);
    }
    template<typename... Args>
    void assign(Args&&... args)
    {
        v.assign(std::forward<Args>(args)...);
    }
    void shrink_to_fit() { v.shrink_to_fit(); }

    // unqualified erase_if(v, pred) works for std::vector and Vector alike
    template<typename Pred>
    friend auto erase_if(Vector& vec, Pred pred)
    {
        return std::erase_if(vec.v, pred);
    }

protected:
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
            r.copy_values_from(*this);
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
