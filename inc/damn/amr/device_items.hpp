#ifndef DAMN_AMR_DEVICE_ITEMS_HPP
#define DAMN_AMR_DEVICE_ITEMS_HPP

// a copy's leaving items in unified memory, moved on the device as columns, the host never
//  touching them: doc/amr/device_items.md

#include "damn/core/def/damn_config.hpp"

#if DAMN_HAVE_GPU

#include "damn/amr/cell_item_data.hpp"
#include "damn/amr/device_copy.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/core/def/detail/mkn_gpu.hpp"
#include "damn/core/exec/kernels.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "mkn/gpu/scan.hpp"

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace damn::amr
{
inline constexpr std::size_t max_item_columns = 16;

// the bytes of one value of each of a resource's columns, the cell columns left out
struct ItemColumns
{
    std::array<std::uint8_t, max_item_columns> bytes{};
    std::uint8_t n = 0;

    bool operator==(ItemColumns const&) const = default;
};

// a source's items, grouped by cell: cell c's (C order over cells) are [starts[c], starts[c] +
//  sizes[c]) in every column
template<std::size_t dim>
struct DeviceItems
{
    core::Box<int, dim> cells{};
    std::uint32_t const* starts = nullptr;
    std::uint32_t const* sizes  = nullptr;
    std::array<std::byte const*, max_item_columns> columns{};
    ItemColumns layout{};
};

// a destination's new items, from the first: their columns, and their cell (a column per
//  direction), written from their cell's place in the copy
template<std::size_t dim>
struct DeviceItemsOut
{
    std::array<std::byte*, max_item_columns> columns{};
    ItemColumns layout{};
    std::array<int*, dim> cell{};
};

// a resource's hooks (ADL): device_items(src) throws if its items are not grouped by cell;
//  device_items_appended(dst, n) grows dst by n items, cell_items::appended follows the copy
template<typename R, std::size_t dim>
concept DeviceItemSource = detail::OnDevice<R> and requires(R const& r) {
    { device_items(r) } -> std::same_as<DeviceItems<dim>>;
};

template<typename R, std::size_t dim>
concept DeviceItemDestination = detail::OnDevice<R> and requires(R& r, std::size_t const n) {
    { device_items_appended(r, n) } -> std::same_as<DeviceItemsOut<dim>>;
};

namespace detail
{
    // a piece's block in a buffer: its cells' item offsets (CSR, cells + 1), then each column,
    //  8 byte aligned
    struct ItemBlock
    {
        std::size_t columns[max_item_columns]{}; // from the block's start
        std::size_t bytes = 0;
    };

    inline std::size_t aligned(std::size_t const bytes)
    {
        return (bytes + 7) / 8 * 8;
    }

    inline ItemBlock item_block(std::size_t const cells, std::size_t const items,
                                ItemColumns const& layout)
    {
        ItemBlock block;
        block.bytes = aligned((cells + 1) * sizeof(std::uint32_t));
        for (std::size_t c = 0; c < layout.n; ++c)
        {
            block.columns[c] = block.bytes;
            block.bytes += aligned(items * layout.bytes[c]);
        }
        return block;
    }

    // a piece's header in a counts message: its items, its layout (bytes per column, 8 a word)
    inline constexpr std::size_t header_words = 3;

    inline void write_header(std::uint64_t* const at, std::size_t const items,
                             ItemColumns const& layout)
    {
        at[0] = items;
        at[1] = at[2] = 0;
        for (std::size_t c = 0; c < layout.n; ++c)
            at[1 + c / 8] |= std::uint64_t{layout.bytes[c]} << (8 * (c % 8));
    }

    inline std::pair<std::size_t, ItemColumns> read_header(std::uint64_t const* const at)
    {
        ItemColumns layout;
        for (std::size_t c = 0; c < max_item_columns; ++c)
            if (auto const b = (at[1 + c / 8] >> (8 * (c % 8))) & 0xff; b)
                layout.bytes[layout.n++] = static_cast<std::uint8_t>(b);
        return {static_cast<std::size_t>(at[0]), layout};
    }

    template<std::size_t dim>
    struct PackPiece
    {
        std::uint32_t const* starts;
        std::uint32_t const* sizes;
        core::Box<int, dim> src_cells;
        core::Box<int, dim> box; // destination cells
        core::Point<int, dim> shift;
        std::size_t first_cell; // in the launch's cells
        std::size_t first;      // its first item in the launch's items
        std::array<std::byte const*, max_item_columns> src;
        std::array<std::byte*, max_item_columns> dst;
        ItemColumns layout;
        std::uint32_t* offsets;
    };

    template<std::size_t dim>
    struct UnpackPiece
    {
        std::uint32_t const* offsets;
        std::array<std::byte const*, max_item_columns> src;
        std::array<std::byte*, max_item_columns> dst;
        ItemColumns layout;
        std::array<int*, dim> cell;
        core::Box<int, dim> box;
        std::size_t first; // its first item in the launch's items
    };

    struct ItemPeer
    {
        int rank = 0;
        std::vector<std::size_t> pieces{};   // outgoing: its pack indices
        std::vector<std::uint64_t> header{}; // per piece header_words (counts message)
        std::size_t begin = 0, bytes = 0;    // its blocks in the send or receive buffer
    };

    template<std::size_t dim>
    struct ItemSlot // a destination: a patch's resource
    {
        std::size_t piece = 0; // a piece of it, to reach it
        std::size_t total = 0, placed = 0;
        DeviceItemsOut<dim> out{};
    };

    // kept with the plan, buffers grown, never shrunk
    template<std::size_t dim>
    struct ItemTable
    {
        bool usable = false; // false: host resources, the host path

        core::gpu::Vec_t<PackPiece<dim>> pack;
        core::gpu::Vec_t<UnpackPiece<dim>> unpack;
        core::gpu::Vec_t<std::uint64_t> counts, scan, totals;
        std::vector<std::uint64_t> totals_host{};
        core::gpu::Vec_t<std::uint64_t> send, recv; // words: blocks 8 byte aligned

        std::size_t n_local    = 0;    // local pieces, first in pack (plan order)
        std::size_t self_bytes = 0;    // their blocks, first in send
        std::vector<std::size_t> at{}; // per pack piece its block in send
        std::vector<ItemPeer> to{};    // rank order, pieces in outgoing order
        std::vector<ItemPeer> from{};  // as Plan::from
        // per plan piece: its block, its items and their layout, its destination
        std::vector<std::byte const*> block{};
        std::vector<std::size_t> items{};
        std::vector<ItemColumns> layout{};
        std::vector<std::size_t> slot{};
        std::vector<ItemSlot<dim>> slots{};
    };

    template<std::size_t dim, typename R>
        requires DeviceItemSource<R, dim>
    Residence item_source_side(R const&)
    {
        return Residence::Device;
    }
    template<std::size_t dim, typename R>
        requires(!OnDevice<R>)
    Residence item_source_side(R const&)
    {
        return Residence::Host;
    }
    template<std::size_t dim, typename R>
    Residence item_source_side(R const&)
    {
        return Residence::Unsupported;
    }

    template<std::size_t dim, typename R>
        requires DeviceItemDestination<R, dim>
    Residence item_destination_side(R&)
    {
        return Residence::Device;
    }
    template<std::size_t dim, typename R>
        requires(!OnDevice<R>)
    Residence item_destination_side(R&)
    {
        return Residence::Host;
    }
    template<std::size_t dim, typename R>
    Residence item_destination_side(R&)
    {
        return Residence::Unsupported;
    }

    template<std::size_t dim, typename R>
        requires DeviceItemSource<R, dim>
    DeviceItems<dim> items_of(R const& r)
    {
        return device_items(r);
    }
    template<std::size_t dim>
    DeviceItems<dim> items_of(auto const&)
    {
        throw std::logic_error("damn::amr::copy: a device piece without device items");
    }

    template<std::size_t dim, typename R>
        requires DeviceItemDestination<R, dim>
    DeviceItemsOut<dim> appended_of(R& r, std::size_t const n)
    {
        return device_items_appended(r, n);
    }
    template<std::size_t dim>
    DeviceItemsOut<dim> appended_of(auto&, std::size_t)
    {
        throw std::logic_error("damn::amr::copy: a device piece without a device destination");
    }

    template<std::size_t dim, typename Access, typename PatchData>
    Sides item_plan_sides(Plan<dim> const& plan, Access const& access,
                          LevelData<dim, PatchData>& ld)
    {
        Sides sides;
        for (auto const& piece : plan.pieces)
            dst_of<dim, true>(access, ld.data[piece.dst], piece, [&](auto& dst) {
                sides.add(item_destination_side<dim>(dst), piece.names.dst);
                if (is_local(piece))
                    src_of<dim, true>(access, ld.data[piece.src.local], piece,
                                      [&](auto const& src) {
                                          sides.add(item_source_side<dim>(src), piece.names.src);
                                      });
            });
        for (auto const& piece : plan.outgoing)
            src_of<dim, true>(access, ld.data[piece.src.local], piece, [&](auto const& src) {
                sides.add(item_source_side<dim>(src), piece.names.src);
            });
        return sides;
    }

    template<std::size_t dim, typename Access, typename PatchData>
    ItemTable<dim>& item_table(Plan<dim>& plan, Access const& access, LevelData<dim, PatchData>& ld)
    {
        if (plan.device)
            return *static_cast<ItemTable<dim>*>(plan.device.get());
        auto table    = std::make_shared<ItemTable<dim>>();
        table->usable = item_plan_sides(plan, access, ld).device > 0;
        plan.device   = table;

        // what the plan fixes: pieces per peer, destinations
        for (auto const& piece : plan.pieces)
            table->n_local += is_local(piece);
        std::map<int, std::vector<std::size_t>> by_peer;
        for (std::size_t j = 0; j < plan.outgoing.size(); ++j)
            by_peer[plan.outgoing[j].dst_rank].push_back(table->n_local + j);
        for (auto& [rank, pieces] : by_peer)
            table->to.push_back({.rank = rank, .pieces = std::move(pieces)});
        for (auto const& [rank, indices] : plan.from)
            table->from.push_back(
                {.rank   = rank,
                 .header = std::vector<std::uint64_t>(indices.size() * header_words)});
        std::map<std::pair<std::size_t, std::uint32_t>, std::size_t> slot_of;
        for (std::size_t i = 0; i < plan.pieces.size(); ++i)
        {
            auto const key    = std::pair{plan.pieces[i].dst, plan.pieces[i].pair};
            auto const [s, _] = slot_of.emplace(key, table->slots.size());
            if (s->second == table->slots.size())
                table->slots.push_back({.piece = i});
            table->slot.push_back(s->second);
        }
        return *table;
    }

    // the last k in [lo, hi) with key(k) <= i (key(lo) <= i)
    template<typename Key>
    std::size_t last_at_most(std::size_t lo, std::size_t hi, std::size_t const i,
                             Key const& key) DAMN_ALL
    {
        while (hi - lo > 1)
            if (auto const mid = (lo + hi) / 2; key(mid) <= i)
                lo = mid;
            else
                hi = mid;
        return lo;
    }

    inline void copy_value(std::byte* const to, std::byte const* const from,
                           std::uint8_t const bytes) DAMN_ALL
    {
        // constant sizes: one load and store for the usual ones
        switch (bytes)
        {
            case 8: __builtin_memcpy(to, from, 8); return;
            case 4: __builtin_memcpy(to, from, 4); return;
            default:
                for (std::uint8_t b = 0; b < bytes; ++b)
                    to[b] = from[b];
        }
    }
} // namespace detail

// the leaving items of the plan's local and outgoing pieces packed into the table's send
//  buffer: the local pieces' blocks first, then per peer (rank order) a counts message (host)
//  and its blocks
template<std::size_t dim, typename Access, typename PatchData>
void pack_items_on_device(Plan<dim> const& plan, Access const& access,
                          LevelData<dim, PatchData>& ld, detail::ItemTable<dim>& table)
{
    using namespace detail;
    auto& pack = table.pack;
    pack.clear();

    // sources: the local pieces (plan order), then the outgoing ones (plan order)
    std::size_t cells = 0;
    auto const add    = [&](Piece<dim> const& piece) {
        src_of<dim, true>(access, ld.data[piece.src.local], piece, [&](auto const& src) {
            auto const items = items_of<dim>(src);
            pack.push_back({items.starts,
                            items.sizes,
                            items.cells,
                            piece.box,
                            piece.src.shift,
                            cells,
                            0,
                            items.columns,
                            {},
                            items.layout,
                            nullptr});
            cells += piece.box.size();
        });
    };
    for (auto const& piece : plan.pieces)
        if (is_local(piece))
            add(piece);
    for (auto const& piece : plan.outgoing)
        add(piece);
    auto const n_local = table.n_local;

    // per source cell its items, their scan, per piece its items
    table.counts.resize(cells);
    table.scan.resize(cells);
    table.totals.resize(pack.size());
    table.totals_host.resize(pack.size());
    if (cells)
    {
        auto const* const pieces = pack.data();
        auto const n             = pack.size();
        auto* const counts       = table.counts.data();
        kernel::launch(cells, [=] DAMN_ALL() {
            auto const i = kernel::idx();
            auto const& pc
                = pieces[last_at_most(0, n, i, [&](auto const k) { return pieces[k].first_cell; })];
            auto const p = point_in(pc.box, i - pc.first_cell);
            core::Point<int, dim> q;
            for (std::size_t d = 0; d < dim; ++d)
                q[d] = p[d] - pc.shift[d];
            counts[i] = contains(pc.src_cells, q) ? pc.sizes[flat_in(pc.src_cells, q)] : 0;
        });
        (void)mkn::gpu::exclusive_scan(counts, table.scan.data(), cells);
        auto const* const scan = table.scan.data();
        auto* const totals     = table.totals.data();
        kernel::launch(n, [=] DAMN_ALL() {
            auto const k    = kernel::idx();
            auto const last = pieces[k].first_cell + pieces[k].box.size() - 1;
            totals[k]       = scan[last] + counts[last] - scan[pieces[k].first_cell];
        });
        mkn::gpu::take(table.totals.data(), table.totals_host.data(), pack.size());
    }

    // blocks: local ones, then per peer
    auto& at = table.at;
    at.resize(pack.size());
    std::size_t bytes = 0, items = 0;
    for (std::size_t k = 0; k < pack.size(); ++k)
    {
        pack[k].first = items;
        items += table.totals_host[k];
    }
    for (std::size_t k = 0; k < n_local; ++k)
    {
        at[k] = bytes;
        bytes += item_block(pack[k].box.size(), table.totals_host[k], pack[k].layout).bytes;
    }
    table.self_bytes = bytes;

    for (auto& peer : table.to)
    {
        peer.begin = bytes;
        peer.header.resize(peer.pieces.size() * header_words);
        for (std::size_t j = 0; j < peer.pieces.size(); ++j)
        {
            auto const k = peer.pieces[j];
            write_header(peer.header.data() + j * header_words, table.totals_host[k],
                         pack[k].layout);
            at[k] = bytes;
            bytes += item_block(pack[k].box.size(), table.totals_host[k], pack[k].layout).bytes;
        }
        peer.bytes = bytes - peer.begin;
    }

    table.send.resize(bytes / 8);
    auto* const send = reinterpret_cast<std::byte*>(table.send.data());
    for (std::size_t k = 0; k < pack.size(); ++k)
    {
        auto const block = item_block(pack[k].box.size(), table.totals_host[k], pack[k].layout);
        pack[k].offsets  = reinterpret_cast<std::uint32_t*>(send + at[k]);
        for (std::size_t c = 0; c < pack[k].layout.n; ++c)
            pack[k].dst[c] = send + at[k] + block.columns[c];
    }

    // the local pieces' blocks, read by unpack_items_on_device
    table.block.assign(plan.pieces.size(), nullptr); // kept: no allocation
    table.items.assign(plan.pieces.size(), 0);
    table.layout.assign(plan.pieces.size(), {});
    for (std::size_t i = 0, k = 0; i < plan.pieces.size(); ++i)
        if (is_local(plan.pieces[i]))
        {
            table.block[i]  = send + at[k];
            table.items[i]  = table.totals_host[k];
            table.layout[i] = pack[k].layout;
            ++k;
        }

    if (cells == 0)
        return;
    auto const* const pieces = pack.data();
    auto const n             = pack.size();
    auto const* const counts = table.counts.data();
    auto const* const scan   = table.scan.data();
    kernel::launch(cells, [=] DAMN_ALL() { // the blocks' offsets
        auto const i = kernel::idx();
        auto const& pc
            = pieces[last_at_most(0, n, i, [&](auto const k) { return pieces[k].first_cell; })];
        auto const c  = i - pc.first_cell;
        pc.offsets[c] = static_cast<std::uint32_t>(scan[i] - scan[pc.first_cell]);
        if (c + 1 == pc.box.size())
            pc.offsets[c + 1]
                = static_cast<std::uint32_t>(scan[i] + counts[i] - scan[pc.first_cell]);
    });
    if (items == 0)
        return;
    kernel::launch(items, [=] DAMN_ALL() { // a thread per item
        auto const i = kernel::idx();
        auto const& pc
            = pieces[last_at_most(0, n, i, [&](auto const k) { return pieces[k].first; })];
        auto const j  = i - pc.first;
        auto const fc = pc.first_cell;
        // its cell: the last with an offset <= j (the first of equal offsets is empty)
        auto const g = last_at_most(fc, fc + pc.box.size(), j,
                                    [&](auto const c) { return scan[c] - scan[fc]; });
        auto const p = point_in(pc.box, g - fc);
        core::Point<int, dim> q;
        for (std::size_t d = 0; d < dim; ++d)
            q[d] = p[d] - pc.shift[d];
        auto const from = pc.starts[flat_in(pc.src_cells, q)] + (j - (scan[g] - scan[fc]));
        for (std::size_t c = 0; c < pc.layout.n; ++c)
            copy_value(pc.dst[c] + j * pc.layout.bytes[c], pc.src[c] + from * pc.layout.bytes[c],
                       pc.layout.bytes[c]);
    });
}

namespace detail
{
    // the bytes of peer k's blocks (table.from), from its counts message
    template<std::size_t dim>
    std::size_t received_bytes(Plan<dim> const& plan, ItemTable<dim> const& table,
                               std::size_t const k)
    {
        auto const& indices = plan.from[k].second;
        std::size_t bytes   = 0;
        for (std::size_t j = 0; j < indices.size(); ++j)
        {
            auto const [items, layout]
                = read_header(table.from[k].header.data() + j * header_words);
            bytes += item_block(plan.pieces[indices[j]].box.size(), items, layout).bytes;
        }
        return bytes;
    }
} // namespace detail

// the blocks of the counts message from peer (rank order in table.from, its pieces in plan
//  order) at their place in the receive buffer: false if their sizes are not the peer's bytes
template<std::size_t dim>
bool place_received(Plan<dim> const& plan, detail::ItemTable<dim>& table, std::size_t const k)
{
    using namespace detail;
    auto& peer           = table.from[k];
    auto const& indices  = plan.from[k].second;
    auto const* const at = reinterpret_cast<std::byte const*>(table.recv.data()) + peer.begin;
    std::size_t bytes    = 0;
    for (std::size_t j = 0; j < indices.size(); ++j)
    {
        auto const i               = indices[j];
        auto const [items, layout] = read_header(peer.header.data() + j * header_words);
        table.block[i]             = at + bytes;
        table.items[i]             = items;
        table.layout[i]            = layout;
        bytes += item_block(plan.pieces[i].box.size(), items, layout).bytes;
    }
    return bytes == peer.bytes;
}

// every plan piece's block (table.block) appended to its destination, then cell_items::appended
//  per destination is the caller's
template<std::size_t dim, typename Access, typename PatchData>
void unpack_items_on_device(Plan<dim> const& plan, Access const& access,
                            LevelData<dim, PatchData>& ld, detail::ItemTable<dim>& table)
{
    using namespace detail;
    auto const& pieces = plan.pieces;

    // per destination (patch, resource) its new items, grown once
    for (auto& slot : table.slots)
        slot.total = slot.placed = 0;
    for (std::size_t i = 0; i < pieces.size(); ++i)
        table.slots[table.slot[i]].total += table.items[i];
    for (auto& slot : table.slots)
        if (slot.total)
        {
            auto const& piece = pieces[slot.piece];
            dst_of<dim, true>(access, ld.data[piece.dst], piece,
                              [&](auto& dst) { slot.out = appended_of<dim>(dst, slot.total); });
        }

    auto& unpack = table.unpack;
    unpack.clear();
    std::size_t items = 0;
    for (std::size_t i = 0; i < pieces.size(); ++i)
    {
        auto const& piece = pieces[i];
        if (table.items[i] == 0)
            continue;
        auto& slot = table.slots[table.slot[i]];
        if (slot.out.layout != table.layout[i])
            throw std::runtime_error("damn::amr::copy: " + piece.names.src + " to "
                                     + piece.names.dst + ": the items' columns differ");
        auto const block = item_block(piece.box.size(), table.items[i], table.layout[i]);
        UnpackPiece<dim> up{reinterpret_cast<std::uint32_t const*>(table.block[i]),
                            {},
                            {},
                            table.layout[i],
                            {},
                            piece.box,
                            items};
        for (std::size_t c = 0; c < up.layout.n; ++c)
        {
            up.src[c] = table.block[i] + block.columns[c];
            up.dst[c] = slot.out.columns[c] + slot.placed * up.layout.bytes[c];
        }
        for (std::size_t d = 0; d < dim; ++d)
            up.cell[d] = slot.out.cell[d] + slot.placed;
        unpack.push_back(up);
        slot.placed += table.items[i];
        items += table.items[i];
    }
    if (items == 0)
        return;

    auto const* const ups = unpack.data();
    auto const n          = unpack.size();
    kernel::launch(items, [=] DAMN_ALL() { // a thread per item
        auto const i   = kernel::idx();
        auto const& up = ups[last_at_most(0, n, i, [&](auto const k) { return ups[k].first; })];
        auto const j   = i - up.first;
        auto const c   = last_at_most(0, up.box.size(), j,
                                      [&](auto const k) { return std::size_t{up.offsets[k]}; });
        auto const p   = point_in(up.box, c);
        for (std::size_t d = 0; d < dim; ++d)
            up.cell[d][j] = p[d];
        for (std::size_t k = 0; k < up.layout.n; ++k)
            copy_value(up.dst[k] + j * up.layout.bytes[k], up.src[k] + j * up.layout.bytes[k],
                       up.layout.bytes[k]);
    });
}

// a copy of leaving items on one rank on the device: false (nothing done) for host resources
template<std::size_t dim, typename Access, typename PatchData>
bool items_on_device(Plan<dim>& plan, Access const& access, LevelData<dim, PatchData>& ld)
{
    auto& table = detail::item_table(plan, access, ld);
    if (!table.usable)
        return false;
    pack_items_on_device(plan, access, ld, table);
    unpack_items_on_device(plan, access, ld, table);
    return true;
}

} // namespace damn::amr

#endif // DAMN_HAVE_GPU

#endif /* DAMN_AMR_DEVICE_ITEMS_HPP */
