#ifndef DAMN_AMR_CELL_ITEMS_HPP
#define DAMN_AMR_CELL_ITEMS_HPP

// moving a CellItemData resource's items between patches: the items whose cell + shift is in a
// box, packed to bytes and unpacked (other ranks, regrid transfers), or appended directly (the
// same rank). A resource can do each its own way, through these optional free functions (ADL),
// else damn does it item by item (for_each_item, cell, append, the item's bytes):
//
//   void pack_items(src const&, box, shift, std::vector<std::byte>&)
//                                  // appends the selected items in the resource's own format,
//                                  //  e.g. a count then one byte stream per component
//   void unpack_items(dst&, std::byte const*& at, shift)
//                                  // appends what pack_items wrote, cells moved by shift, at
//                                  //  past it
//   std::vector<value_type> read_items(dst const&, std::byte const*& at)
//                                  // what pack_items wrote as items, at past it (refine's
//                                  //  coarse items)
//   void append_items_from(dst&, src const&, box, shift)
//                                  // the same rank: directly (src may be dst); without it a
//                                  //  resource with pack_items goes through a scratch buffer
//   void items_appended(dst&)      // after a copy or a regrid appended to dst (once, whatever
//                                  //  the number of pieces), e.g. to sort it
//
// pack_items and unpack_items come together; the default format: a count, then the items' bytes

#include "damn/core/memory/scratch.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace damn::core
{
template<typename Type, std::size_t dim>
class Point;
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core

namespace damn::amr::cell_items
{
namespace detail
{
    template<typename T>
    void write(std::vector<std::byte>& out, T const& v)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        auto const at = out.size();
        out.resize(at + sizeof(T));
        std::memcpy(out.data() + at, &v, sizeof(T));
    }

    template<typename T>
    T read(std::byte const*& at)
    {
        T v;
        std::memcpy(&v, at, sizeof(T));
        at += sizeof(T);
        return v;
    }

    template<typename R, std::size_t dim>
    concept Packs = requires(R& r, R const& cr, core::Box<int, dim> const& box,
                             core::Point<int, dim> const& shift, std::vector<std::byte>& out,
                             std::byte const*& at) {
        pack_items(cr, box, shift, out);
        unpack_items(r, at, shift);
    };
} // namespace detail

// src's items whose cell + shift is in box, appended to out
template<std::size_t dim, typename R>
void pack(R const& src, core::Box<int, dim> const& box, core::Point<int, dim> const& shift,
          std::vector<std::byte>& out)
{
    if constexpr (detail::Packs<R, dim>)
        pack_items(src, box, shift, out);
    else
    {
        auto const count_at = out.size();
        detail::write(out, std::uint64_t{0});
        std::uint64_t count = 0;
        for_each_item(src, [&](auto const& item) {
            if (isIn(cell(item) + shift, box))
            {
                detail::write(out, item);
                ++count;
            }
        });
        std::memcpy(out.data() + count_at, &count, sizeof(count));
    }
}

// what pack wrote, appended to dst with its cells moved by shift; at moves past it
template<std::size_t dim, typename R>
void unpack(R& dst, std::byte const*& at, core::Point<int, dim> const& shift)
{
    if constexpr (detail::Packs<R, dim>)
        unpack_items(dst, at, shift);
    else
    {
        using Item       = typename R::value_type;
        auto const count = detail::read<std::uint64_t>(at);
        for (std::uint64_t i = 0; i < count; ++i)
            append(dst, detail::read<Item>(at), shift);
    }
}

// what pack wrote, as items (their cells as they were); at moves past it
template<typename R>
std::vector<typename R::value_type> read(R const& like, std::byte const*& at)
{
    using Item = typename R::value_type;
    if constexpr (requires { read_items(like, at); })
        return read_items(like, at);
    else
    {
        auto const count = detail::read<std::uint64_t>(at);
        std::vector<Item> items(count);
        if (count)
            std::memcpy(items.data(), at, count * sizeof(Item));
        at += count * sizeof(Item);
        return items;
    }
}

// src's items whose cell + shift is in box appended to dst (the same rank; src may be dst).
//  src may be another resource type of the same items, e.g. a read only handle of them
template<std::size_t dim, typename R, typename S>
void append_from(R& dst, S const& src, core::Box<int, dim> const& box,
                 core::Point<int, dim> const& shift)
{
    if constexpr (requires { append_items_from(dst, src, box, shift); })
        append_items_from(dst, src, box, shift);
    else if constexpr (detail::Packs<R, dim> and detail::Packs<S, dim>)
    {
        core::ScratchVector<std::byte> bytes; // packed first: appending to src itself is safe
        pack_items(src, box, shift, *bytes);
        std::byte const* at = bytes->data();
        unpack_items(dst, at, shift);
    }
    else
    {
        using Item          = typename R::value_type;
        auto const selected = [&](auto const& item) { return isIn(cell(item) + shift, box); };
        if (static_cast<void const*>(&src) != static_cast<void const*>(&dst))
        {
            for_each_item(src, [&](auto const& item) {
                if (selected(item))
                    append(dst, item, shift);
            });
            return;
        }
        // the same array: collect first, appending invalidates it
        core::ScratchVector<Item> items;
        for_each_item(src, [&](auto const& item) {
            if (selected(item))
                items->push_back(item);
        });
        for (auto const& item : *items)
            append(dst, item, shift);
    }
}

// after appending to dst (a copy, a regrid), once
template<typename R>
void appended(R& dst)
{
    if constexpr (requires { items_appended(dst); })
        items_appended(dst);
}

} // namespace damn::amr::cell_items

#endif // DAMN_AMR_CELL_ITEMS_HPP
