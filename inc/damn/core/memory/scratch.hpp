#ifndef DAMN_CORE_MEMORY_SCRATCH_HPP
#define DAMN_CORE_MEMORY_SCRATCH_HPP

// per thread scratch for hot paths instead of allocating: a holder takes a buffer from its
//  thread's slots for its scope. A nested use on the same thread (recursion, a hook calling back)
//  takes the next slot, never one still in use, so uses need no names and cannot overwrite each
//  other. The buffers are MinimizingVectors: capacity is kept between uses (no allocation once
//  warm) and shrinks after a run of uses well under it
//
//   Scratch<T> s{n};       n values, left from earlier uses (not zeroed)
//   ScratchVector<T> v;    an empty std::vector, for buffers grown by push_back

#include "damn/core/vector.hpp"
#include "damn/core/def/damn_config.hpp"

#include <cstddef>
#include <deque>
#include <span>
#include <utility>

namespace damn::core
{
namespace detail
{
    // a thread's buffers of one type, a stack: a holder takes the next, gives it back on scope
    //  exit (a deque: a slot stays put as nested ones are added)
    template<typename Kept>
    struct ScratchSlots
    {
        std::deque<Kept> slots;
        std::size_t depth = 0;

        static ScratchSlots& mine()
        {
            thread_local ScratchSlots s;
            return s;
        }
        Kept& take()
        {
            if (slots.size() <= depth)
                slots.emplace_back();
            return slots[depth++];
        }
        void give() { --depth; }
    };
} // namespace detail

template<typename T, auto alloc_mode = AllocatorMode::CPU>
class Scratch
{
    using Kept  = MinimizingVector<T, alloc_mode>;
    using Slots = detail::ScratchSlots<Kept>;

public:
    explicit Scratch(std::size_t const n)
        : kept{Slots::mine().take()}
    {
        kept.template get<false>(n);
    }
    Scratch(Scratch const&)            = delete;
    Scratch& operator=(Scratch const&) = delete;
    ~Scratch() { Slots::mine().give(); }

    auto data() { return kept().data(); }
    auto size() const { return kept.size(); }
    auto begin() { return kept().begin(); }
    auto end() { return kept().end(); }
    auto& operator[](std::size_t const i) { return kept[i]; }
    std::span<T> span() { return {kept().data(), kept.size()}; }

private:
    Kept& kept;
};

template<typename T, auto alloc_mode = AllocatorMode::CPU>
class ScratchVector
{
    using Kept  = MinimizingVector<T, alloc_mode>;
    using Slots = detail::ScratchSlots<Kept>;

public:
    ScratchVector()
        : kept{Slots::mine().take()}
    {
        kept.clear();
    }
    ScratchVector(ScratchVector const&)            = delete;
    ScratchVector& operator=(ScratchVector const&) = delete;
    ~ScratchVector()
    {
        kept.template fit<false>(kept.size());
        kept.clear();
        Slots::mine().give();
    }

    auto data() { return kept().data(); }
    auto size() const { return kept.size(); }
    bool empty() const { return kept.size() == 0; }
    auto begin() { return kept().begin(); }
    auto end() { return kept().end(); }
    auto& operator[](std::size_t const i) { return kept[i]; }
    void resize(std::size_t const n) { kept().resize(n); }
    void push_back(T const& t) { kept().push_back(t); }
    auto& emplace_back(auto&&... args)
    {
        return kept().emplace_back(std::forward<decltype(args)>(args)...);
    }
    // the vector itself, e.g. for a hook appending to a std::vector&: a hook's parameter must be
    //  that type, a deduced one (auto&) gets the ScratchVector
    auto& operator*() { return kept(); }
    operator typename Kept::vector_t&() { return kept(); }

private:
    Kept& kept;
};

} // namespace damn::core

#endif // DAMN_CORE_MEMORY_SCRATCH_HPP
