#ifndef DAMN_AMR_DEVICE_COPY_HPP
#define DAMN_AMR_DEVICE_COPY_HPP

// copies of resources in unified memory on the device, the host never touching their values:
//  doc/amr/device_copy.md. Pieces of another rank's go through the table's send and receive
//  buffers (damn::mpi::copy). A device resource the device copy cannot take throws

#include "damn/core/def/damn_config.hpp"

#if DAMN_HAVE_GPU

#include "damn/amr/box_data.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/core/exec/kernels.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>

namespace damn::amr
{

template<typename R, std::size_t dim>
concept DeviceBoxSpan
    = BoxData<std::remove_cvref_t<R>, dim> and requires(R& r, core::Box<int, dim> const& box) {
          requires std::remove_cvref_t<R>::on_device;
          { box_span(r, box).ptr };
          { box_span(r, box).box };
      };

namespace detail
{
    template<std::size_t dim>
    struct DevicePiece
    {
        double const* src;
        core::Box<int, dim> src_box; // src's points (its span's box)
        double* dst;
        core::Box<int, dim> dst_box;
        core::Box<int, dim> box; // the piece's (destination) points
        core::Point<int, dim> shift;
        std::size_t first; // its first point in the launch's range
    };

    // the pieces of a border sum or max landing on one destination and resource pair
    template<std::size_t dim>
    struct BorderGroup
    {
        double* dst;
        core::Box<int, dim> dst_box;
        core::Box<int, dim> bounds; // the group's pieces' boxes' bounding box
        std::size_t first;          // its first point in the launch's range
        std::size_t begin, end;     // its contributors
    };

    // a peer's message: values [begin, begin + count) of the send or receive buffer
    struct DevicePeer
    {
        int rank;
        std::size_t begin, count;
    };

    template<std::size_t dim>
    struct DeviceTable
    {
        bool usable       = false; // false: host resources, the host path
        std::size_t total = 0;
        // per plan piece (plan order): a local one from its source, another rank's from its
        //  values in recv
        core::gpu::Vec_t<DevicePiece<dim>> pieces;

        // messages, in plan order per peer as the host path's (the same bytes): pieces of
        //  outgoing packed into send, peers in rank order in both buffers
        std::size_t total_out = 0;
        core::gpu::Vec_t<DevicePiece<dim>> outgoing;
        core::gpu::Vec_t<double> send, recv;
        std::vector<DevicePeer> to, from;

        // sums and maxes (built on the first): the snapshot, and per (destination, resources)
        //  group its points' bounds and its contributors in order (a piece's index, own: -1)
        bool groups_built = false;
        core::gpu::Vec_t<double> snapshot;
        core::gpu::Vec_t<BorderGroup<dim>> groups;
        core::gpu::Vec_t<long> contributors;
        std::size_t group_points = 0;
    };

    // flat C order index of p in box
    template<std::size_t dim>
    std::size_t flat_in(core::Box<int, dim> const& box, core::Point<int, dim> const& p) DAMN_ALL
    {
        std::size_t at = 0;
        for (std::size_t d = 0; d < dim; ++d)
            at = at * static_cast<std::size_t>(box.upper[d] - box.lower[d] + 1)
                 + static_cast<std::size_t>(p[d] - box.lower[d]);
        return at;
    }
    // the point of flat C order index i in box
    template<std::size_t dim>
    core::Point<int, dim> point_in(core::Box<int, dim> const& box, std::size_t i) DAMN_ALL
    {
        core::Point<int, dim> p;
        for (std::size_t d = dim; d-- > 0;)
        {
            auto const n = static_cast<std::size_t>(box.upper[d] - box.lower[d] + 1);
            p[d]         = box.lower[d] + static_cast<int>(i % n);
            i /= n;
        }
        return p;
    }

    template<typename R>
    concept OnDevice = requires { requires std::remove_cvref_t<R>::on_device; };

    // the device kernels' resources: a source read through a pointer to (const) double, a
    //  destination written through a pointer to double
    template<typename R, std::size_t dim>
    concept DeviceSource
        = DeviceBoxSpan<R const, dim> and requires(R const& r, core::Box<int, dim> const& box) {
              requires std::same_as<
                  std::remove_cv_t<std::remove_pointer_t<decltype(box_span(r, box).ptr)>>, double>;
          };

    template<typename R, std::size_t dim>
    concept DeviceDestination
        = DeviceBoxSpan<R, dim> and requires(R& r, core::Box<int, dim> const& box) {
              requires std::same_as<decltype(box_span(r, box).ptr), double*>;
          };

    // a side of a piece: on the device and taken by the kernels, on the host, or on the device
    //  and not taken (an error: the host never touches device values)
    enum class Residence : std::uint8_t { Device, Host, Unsupported };

    template<std::size_t dim, typename R>
        requires DeviceSource<R, dim>
    Residence source_side(R const&)
    {
        return Residence::Device;
    }
    template<std::size_t dim, typename R>
        requires(!OnDevice<R>)
    Residence source_side(R const&)
    {
        return Residence::Host;
    }
    template<std::size_t dim, typename R>
    Residence source_side(R const&)
    {
        return Residence::Unsupported;
    }

    template<std::size_t dim, typename R>
        requires DeviceDestination<R, dim>
    Residence destination_side(R&)
    {
        return Residence::Device;
    }
    template<std::size_t dim, typename R>
        requires(!OnDevice<R>)
    Residence destination_side(R&)
    {
        return Residence::Host;
    }
    template<std::size_t dim, typename R>
    Residence destination_side(R&)
    {
        return Residence::Unsupported;
    }

    // every side of a plan the same, the device's or the host's
    struct Sides
    {
        std::size_t device = 0, host = 0;

        void add(Residence const side, std::string const& resource)
        {
            if (side == Residence::Unsupported)
                throw std::runtime_error("damn::amr::copy: " + resource
                                         + " is on the device, the device copy cannot take it"
                                           " (a double box_span), the host must not");
            ++(side == Residence::Device ? device : host);
            if (device and host)
                throw std::runtime_error("damn::amr::copy: " + resource
                                         + " mixes device and host resources in one copy");
        }
    };

    template<std::size_t dim, typename S>
        requires DeviceSource<S, dim>
    void add_source(core::gpu::Vec_t<DevicePiece<dim>>& pieces, std::size_t& total, S const& src,
                    core::Box<int, dim> const& src_patch_box, double* const dst,
                    core::Box<int, dim> const& dst_box, auto const& piece)
    {
        auto const s = box_span(src, src_patch_box);
        pieces.push_back(
            DevicePiece<dim>{s.ptr, s.box, dst, dst_box, piece.box, piece.src.shift, total});
        total += piece.box.size();
    }
    template<std::size_t dim>
    void add_source(core::gpu::Vec_t<DevicePiece<dim>>&, std::size_t&, auto const&,
                    core::Box<int, dim> const&, double*, core::Box<int, dim> const&, auto const&)
    {
        throw std::logic_error("damn::amr::copy: a device piece without a device source");
    }

    template<std::size_t dim, typename D>
        requires DeviceDestination<D, dim>
    DevicePiece<dim> with_destination(DevicePiece<dim> piece, D& dst,
                                      core::Box<int, dim> const& dst_patch_box)
    {
        auto const d  = box_span(dst, dst_patch_box);
        piece.dst     = d.ptr;
        piece.dst_box = d.box;
        return piece;
    }
    template<std::size_t dim>
    DevicePiece<dim> with_destination(DevicePiece<dim>, auto&, core::Box<int, dim> const&)
    {
        throw std::logic_error("damn::amr::copy: a device piece without a device destination");
    }

    // per peer rank (rank order) its values, placed one after the other
    inline void place_peers(std::map<int, std::size_t> const& counts,
                            std::vector<DevicePeer>& peers)
    {
        std::size_t at = 0;
        for (auto const& [rank, count] : counts)
        {
            peers.push_back({rank, at, count});
            at += count;
        }
    }

    inline std::size_t peer_at(std::vector<DevicePeer> const& peers, int const rank)
    {
        for (std::size_t k = 0; k < peers.size(); ++k)
            if (peers[k].rank == rank)
                return k;
        throw std::logic_error("damn::amr::copy: a piece of no peer");
    }

    template<std::size_t dim, typename Access, typename PatchData>
    Sides plan_sides(Plan<dim> const& plan, Access const& access, LevelData<dim, PatchData>& ld)
    {
        Sides sides;
        for (auto const& piece : plan.pieces)
            dst_of<dim, false>(access, ld.data[piece.dst], piece, [&](auto& dst) {
                sides.add(destination_side<dim>(dst), piece.names.dst);
                if (is_local(piece))
                    src_of<dim, false>(access, ld.data[piece.src.local], piece,
                                       [&](auto const& src) {
                                           sides.add(source_side<dim>(src), piece.names.src);
                                       });
            });
        for (auto const& piece : plan.outgoing)
            src_of<dim, false>(access, ld.data[piece.src.local], piece, [&](auto const& src) {
                sides.add(source_side<dim>(src), piece.names.src);
            });
        return sides;
    }

    // the pieces of a plan whose resources are all the device's
    template<std::size_t dim, typename Access, typename PatchData>
    void fill_table(DeviceTable<dim>& table, Plan<dim> const& plan, Access const& access,
                    LevelData<dim, PatchData>& ld)
    {
        std::map<int, std::size_t> out, in;
        for (auto const& piece : plan.outgoing)
            out[piece.dst_rank] += piece.box.size();
        for (auto const& piece : plan.pieces)
            if (!is_local(piece))
                in[piece.src.patch.rank] += piece.box.size();
        place_peers(out, table.to);
        place_peers(in, table.from);
        table.send.resize(table.to.empty() ? 0 : table.to.back().begin + table.to.back().count);
        table.recv.resize(table.from.empty() ? 0
                                             : table.from.back().begin + table.from.back().count);

        std::vector<std::size_t> out_at(table.to.size()), in_at(table.from.size());
        for (std::size_t k = 0; k < table.to.size(); ++k)
            out_at[k] = table.to[k].begin;
        for (std::size_t k = 0; k < table.from.size(); ++k)
            in_at[k] = table.from[k].begin;

        for (auto const& piece : plan.outgoing)
            src_of<dim, false>(access, ld.data[piece.src.local], piece, [&](auto const& src) {
                auto& at = out_at[peer_at(table.to, piece.dst_rank)];
                add_source(table.outgoing, table.total_out, src, piece.src.patch.box,
                           table.send.data() + at, piece.box, piece);
                at += piece.box.size();
            });

        for (auto const& piece : plan.pieces)
            dst_of<dim, false>(access, ld.data[piece.dst], piece, [&](auto& dst) {
                auto const& dst_patch_box = ld.patches[piece.dst].box;
                if (is_local(piece))
                {
                    src_of<dim, false>(access, ld.data[piece.src.local], piece,
                                       [&](auto const& src) {
                                           add_source(table.pieces, table.total, src,
                                                      piece.src.patch.box, nullptr, {}, piece);
                                       });
                    table.pieces.back() = with_destination(table.pieces.back(), dst, dst_patch_box);
                    return;
                }
                // its values, in the piece's points' order
                auto& at = in_at[peer_at(table.from, piece.src.patch.rank)];
                DevicePiece<dim> const received{.src     = table.recv.data() + at,
                                                .src_box = piece.box,
                                                .dst     = nullptr,
                                                .dst_box = {},
                                                .box     = piece.box,
                                                .shift   = {},
                                                .first   = table.total};
                table.pieces.push_back(with_destination(received, dst, dst_patch_box));
                table.total += piece.box.size();
                at += piece.box.size();
            });
    }

    // the plan's table, built on first use
    template<std::size_t dim, typename Access, typename PatchData>
    DeviceTable<dim>& device_table(Plan<dim>& plan, Access const& access,
                                   LevelData<dim, PatchData>& ld)
    {
        if (plan.device)
            return *static_cast<DeviceTable<dim>*>(plan.device.get());
        auto table    = std::make_shared<DeviceTable<dim>>();
        table->usable = plan_sides(plan, access, ld).device > 0;
        if (table->usable)
            fill_table(*table, plan, access, ld);
        plan.device = table;
        return *table;
    }

    template<std::size_t dim>
    bool contains(core::Box<int, dim> const& box, core::Point<int, dim> const& p) DAMN_ALL
    {
        for (std::size_t d = 0; d < dim; ++d)
            if (p[d] < box.lower[d] or p[d] > box.upper[d])
                return false;
        return true;
    }

    // the border groups, in fold_sums' order: pieces by destination and resources, then image
    template<std::size_t dim, typename PatchData>
    void build_groups(Plan<dim> const& plan, DeviceTable<dim>& table,
                      LevelData<dim, PatchData> const& ld)
    {
        auto const& pieces = plan.pieces;
        auto const image   = [&](std::size_t const i) {
            return core::shift(pieces[i].src.patch.box, pieces[i].src.shift);
        };
        std::vector<std::size_t> order(pieces.size());
        for (std::size_t i = 0; i < order.size(); ++i)
            order[i] = i;
        std::sort(order.begin(), order.end(), [&](auto const i, auto const j) {
            if (pieces[i].dst != pieces[j].dst)
                return pieces[i].dst < pieces[j].dst;
            if (pieces[i].names != pieces[j].names)
                return pieces[i].names < pieces[j].names;
            return core::lexicographic_less(image(i), image(j));
        });
        table.snapshot.resize(table.total);
        for (std::size_t begin = 0; begin < order.size();)
        {
            auto const& first = pieces[order[begin]];
            auto end          = begin + 1;
            while (end < order.size() and pieces[order[end]].dst == first.dst
                   and pieces[order[end]].names == first.names)
                ++end;
            auto const& dst_box = table.pieces[order[begin]].dst_box;
            auto const& cells   = ld.patches[first.dst].box;
            BorderGroup<dim> g{
                table.pieces[order[begin]].dst, dst_box, first.box, table.group_points,
                table.contributors.size(),      0};
            for (auto k = begin + 1; k < end; ++k)
                for (std::size_t d = 0; d < dim; ++d)
                {
                    g.bounds.lower[d] = std::min(g.bounds.lower[d], pieces[order[k]].box.lower[d]);
                    g.bounds.upper[d] = std::max(g.bounds.upper[d], pieces[order[k]].box.upper[d]);
                }
            bool own_added = false;
            for (auto k = begin; k < end; ++k)
            {
                if (!own_added and core::lexicographic_less(cells, image(order[k])))
                {
                    table.contributors.push_back(-1);
                    own_added = true;
                }
                table.contributors.push_back(static_cast<long>(order[k]));
            }
            if (!own_added)
                table.contributors.push_back(-1);
            g.end = table.contributors.size();
            table.group_points += g.bounds.size();
            table.groups.push_back(g);
            begin = end;
        }
        table.groups_built = true;
    }

    // the last piece whose first point is <= i
    template<typename Piece>
    std::size_t piece_of(Piece const* const pieces, std::size_t const n,
                         std::size_t const i) DAMN_ALL
    {
        std::size_t lo = 0, hi = n;
        while (hi - lo > 1)
            if (auto const mid = (lo + hi) / 2; pieces[mid].first <= i)
                lo = mid;
            else
                hi = mid;
        return lo;
    }

    // every piece's points, a thread per point: dst(p) = src(p - shift)
    template<std::size_t dim>
    void copy_pieces(core::gpu::Vec_t<DevicePiece<dim>> const& table_pieces,
                     std::size_t const total)
    {
        if (total == 0)
            return;
        auto const* const pieces = table_pieces.data();
        std::size_t const n      = table_pieces.size();
        kernel::launch(total, [=] DAMN_ALL() {
            auto const i   = kernel::idx();
            auto const& pc = pieces[piece_of(pieces, n, i)];
            auto const p   = point_in(pc.box, i - pc.first);
            core::Point<int, dim> q;
            for (std::size_t d = 0; d < dim; ++d)
                q[d] = p[d] - pc.shift[d];
            pc.dst[flat_in(pc.dst_box, p)] = pc.src[flat_in(pc.src_box, q)];
        });
    }

    // sums: from 0 the contributions in fold_sums' order, maxes: from the destination's own
    //  value, std::max's (the first unless less than the next)
    template<Arrival arrival, std::size_t dim>
    void fold_borders(DeviceTable<dim>& table)
    {
        static_assert(arrival == Arrival::Sum or arrival == Arrival::Max);
        auto const* const pieces = table.pieces.data();
        std::size_t const n      = table.pieces.size();
        auto* const snapshot     = table.snapshot.data();
        kernel::launch(table.total, [=] DAMN_ALL() {
            auto const i   = kernel::idx();
            auto const& pc = pieces[piece_of(pieces, n, i)];
            auto const p   = point_in(pc.box, i - pc.first);
            core::Point<int, dim> q;
            for (std::size_t d = 0; d < dim; ++d)
                q[d] = p[d] - pc.shift[d];
            snapshot[i] = pc.src[flat_in(pc.src_box, q)];
        });

        auto const* const groups       = table.groups.data();
        std::size_t const ng           = table.groups.size();
        auto const* const contributors = table.contributors.data();
        kernel::launch(table.group_points, [=] DAMN_ALL() {
            auto const i  = kernel::idx();
            auto const& g = groups[piece_of(groups, ng, i)];
            auto const p  = point_in(g.bounds, i - g.first);
            bool covered  = false;
            for (auto c = g.begin; c < g.end; ++c)
                if (contributors[c] >= 0 and contains(pieces[contributors[c]].box, p))
                    covered = true;
            if (!covered)
                return;
            auto& d          = g.dst[flat_in(g.dst_box, p)];
            double const own = d;
            double acc       = arrival == Arrival::Sum ? 0 : own;
            for (auto c = g.begin; c < g.end; ++c)
            {
                if (contributors[c] < 0)
                {
                    if constexpr (arrival == Arrival::Sum)
                        acc += own;
                    continue;
                }
                auto const& pc = pieces[contributors[c]];
                if (!contains(pc.box, p))
                    continue;
                auto const v = snapshot[pc.first + flat_in(pc.box, p)];
                if constexpr (arrival == Arrival::Sum)
                    acc += v;
                else if (acc < v)
                    acc = v;
            }
            d = acc;
        });
    }
} // namespace detail

// the outgoing pieces into the table's send buffer
template<std::size_t dim>
void pack_on_device(detail::DeviceTable<dim> const& table)
{
    detail::copy_pieces(table.outgoing, table.total_out);
}

// the copy's writes, the other ranks' values in the table's receive buffer
template<Arrival arrival, std::size_t dim, typename PatchData>
void finish_on_device(Plan<dim> const& plan, detail::DeviceTable<dim>& table,
                      LevelData<dim, PatchData> const& ld)
{
    if (table.total == 0)
        return;
    if constexpr (arrival == Arrival::Assign)
        detail::copy_pieces(table.pieces, table.total);
    else
    {
        if (!table.groups_built)
            detail::build_groups(plan, table, ld);
        detail::fold_borders<arrival>(table);
    }
}

// a copy on one rank on the device: false (nothing done) for host resources
template<Arrival arrival, std::size_t dim, typename Access, typename PatchData>
bool copy_on_device(Plan<dim>& plan, Access const& access, LevelData<dim, PatchData>& ld)
{
    auto& table = detail::device_table(plan, access, ld);
    if (!table.usable)
        return false;
    finish_on_device<arrival>(plan, table, ld);
    return true;
}

} // namespace damn::amr

#endif // DAMN_HAVE_GPU

#endif /* DAMN_AMR_DEVICE_COPY_HPP */
