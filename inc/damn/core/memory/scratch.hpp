#ifndef DAMN_CORE_MEMORY_SCRATCH_HPP
#define DAMN_CORE_MEMORY_SCRATCH_HPP

// per thread scratch: hot paths take a span of one of these instead of allocating
//
//   auto own = damn::core::scratch<double, struct fold_own>(n); // std::span<double>, n values
//
// one buffer per thread and (T, Tag), thread_local: pool threads never share one, uses alive at
// the same time on a thread take different tags. A buffer only grows (no allocation once warm),
// its values are left from earlier uses (not zeroed). T: trivially copyable, at most
// max_align_t aligned (the storage's alignment)

#include "damn/core/vector.hpp"
#include "damn/core/def/damn_config.hpp"

#include <cstddef>
#include <deque>
#include <new>
#include <span>
#include <type_traits>

namespace damn::core
{
namespace detail
{
    struct ScratchBuffer
    {
        static constexpr std::align_val_t align{alignof(std::max_align_t)};

        std::byte* data   = nullptr;
        std::size_t bytes = 0;

        ScratchBuffer()                                = default;
        ScratchBuffer(ScratchBuffer const&)            = delete;
        ScratchBuffer& operator=(ScratchBuffer const&) = delete;
        ~ScratchBuffer() { ::operator delete(data, align); }

        std::byte* at_least(std::size_t const n)
        {
            if (n > bytes)
            {
                ::operator delete(data, align);
                data  = nullptr; // if the allocation below throws
                bytes = 0;
                data  = static_cast<std::byte*>(::operator new(n, align));
                bytes = n;
            }
            return data;
        }
    };
} // namespace detail

template<typename T, typename Tag = void>
std::span<T> scratch(std::size_t const n)
{
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(alignof(T) <= alignof(std::max_align_t));
    thread_local detail::ScratchBuffer buffer;
    // operator new's storage: T, an implicit lifetime type, is created in it as used
    return {reinterpret_cast<T*>(buffer.at_least(n * sizeof(T))), n};
}

// a per thread vector, empty, its capacity kept between uses (a MinimizingVector: it shrinks
//  after a run of uses well under it): for buffers grown by push_back, e.g. an ADL hook appending
//  to a std::vector&. A nested use on the same thread (recursion, a hook calling back) takes the
//  next slot, never one still in use
template<typename T, auto alloc_mode = AllocatorMode::CPU>
class ScratchVector
{
    using Kept = MinimizingVector<T, alloc_mode>;

public:
    ScratchVector()
        : kept{take()}
    {
        kept.clear();
    }
    ScratchVector(ScratchVector const&)            = delete;
    ScratchVector& operator=(ScratchVector const&) = delete;
    ~ScratchVector()
    {
        kept.template fit<false>(kept.size());
        kept.clear();
        --depth();
    }

    auto& operator*() { return kept(); }
    auto* operator->() { return &kept(); }

private:
    static std::size_t& depth()
    {
        thread_local std::size_t d = 0;
        return d;
    }
    static Kept& take()
    {
        thread_local std::deque<Kept> slots; // a deque: a slot stays put as nested ones are added
        if (slots.size() <= depth())
            slots.emplace_back();
        return slots[depth()++];
    }

    Kept& kept;
};

} // namespace damn::core

#endif // DAMN_CORE_MEMORY_SCRATCH_HPP
