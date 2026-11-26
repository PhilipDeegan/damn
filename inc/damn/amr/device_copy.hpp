#ifndef DAMN_AMR_DEVICE_COPY_HPP
#define DAMN_AMR_DEVICE_COPY_HPP

// a copy's same rank pieces on the device (GPU builds), for resources in unified memory: their
//  values never touched by the host. A resource qualifies (DeviceBoxSpan) when its type says it
//  is on the device (on_device) and it gives a box_span(resource, patch box): its data and the
//  box of its points (C order). Every local piece of the plan must qualify (else the copy's
//  host path runs), values double
//
//   assign_on_device(plan, access, ld)  Assign: every piece's points in one launch, a thread per
//                                       point (its piece found by binary search over their first
//                                       points), dst(p) = src(p - shift)
//   sum_on_device(plan, access, ld)     Sum: every piece's source values snapshot into a device
//                                       buffer (one launch), then a thread per summed destination
//                                       point adds the contributions in fold_sums' order (from 0,
//                                       the destination's own value at its place among the
//                                       pieces' images): the host path's bits
//
//   append_on_device(plan, access, ld)  Append (items): per resource pair, every piece handed at
//                                       once to the resource type's device hooks (found by ADL):
//                                         can_append_items_on_device(pieces) -> bool
//                                         append_items_on_device(pieces)
//                                       pieces: std::vector<AppendPiece<Dst, Src, dim>>, in plan
//                                       order (a destination's items in the host path's order).
//                                       Nothing done unless every pair can
//
// The pieces' table is built on the plan's first device copy and kept with it (plans live with
//  their level's neighbours: a new decomposition makes new plans, new tables)

#include "damn/core/def/damn_config.hpp"

#if DAMN_HAVE_GPU

#include "damn/amr/box_data.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/core/exec/kernels.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/memory/scratch.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
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

// an Append piece for a resource type's device hook: the resources (copies: handles), the
//  piece's (destination) box and its source's shift, its destination patch (LevelData index)
template<typename Dst, typename Src, std::size_t dim>
struct AppendPiece
{
    Dst dst;
    Src src;
    core::Box<int, dim> box;
    core::Point<int, dim> shift;
    std::size_t dst_patch;
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

    template<std::size_t dim>
    struct SumGroup
    {
        double* dst;
        core::Box<int, dim> dst_box;
        core::Box<int, dim> bounds; // the group's pieces' boxes' bounding box
        std::size_t first;          // its first point in the launch's range
        std::size_t begin, end;     // its contributors
    };

    template<std::size_t dim>
    struct DeviceTable
    {
        bool usable       = false;
        std::size_t total = 0;
        core::gpu::Vec_t<DevicePiece<dim>> pieces;

        // sums (built on the first sum): the snapshot, and per (destination, resources) group
        //  its points' bounds and its contributors in order (a piece's index, own: -1)
        bool sums_built = false;
        core::gpu::Vec_t<double> snapshot;
        core::gpu::Vec_t<SumGroup<dim>> groups;
        core::gpu::Vec_t<long> contributors;
        std::size_t sum_points = 0;
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

    // the assign / sum kernels' resources: a source read through a pointer to (const) double, a
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

    // a piece the kernels copy: into the table
    template<std::size_t dim, typename S, typename D>
        requires DeviceSource<S, dim> and DeviceDestination<D, dim>
    bool table_piece(DeviceTable<dim>& table, S const& src, D& dst, auto const& piece,
                     core::Box<int, dim> const& dst_patch_box)
    {
        auto const s = box_span(src, piece.src.patch.box);
        auto const d = box_span(dst, dst_patch_box);
        table.pieces.push_back(
            DevicePiece<dim>{s.ptr, s.box, d.ptr, d.box, piece.box, piece.src.shift, table.total});
        table.total += piece.box.size();
        return true;
    }

    // other resources: not the device's, the copy's host path
    template<std::size_t dim, typename S, typename D>
    bool table_piece(DeviceTable<dim>&, S const&, D&, auto const&, core::Box<int, dim> const&)
    {
        return false;
    }

    // the plan's table, built on first use: unusable if a local piece's resources do not qualify
    template<std::size_t dim, typename Access, typename PatchData>
    DeviceTable<dim>& device_table(Plan<dim>& plan, Access const& access,
                                   LevelData<dim, PatchData>& ld)
    {
        if (plan.device)
            return *static_cast<DeviceTable<dim>*>(plan.device.get());
        auto table    = std::make_shared<DeviceTable<dim>>();
        table->usable = true;
        for (auto const& piece : plan.pieces)
        {
            if (!is_local(piece))
            {
                table->usable = false;
                break;
            }
            bool qualified = false;
            src_of<dim, false>(access, ld.data[piece.src.local], piece, [&](auto const& src) {
                dst_of<dim, false>(access, ld.data[piece.dst], piece, [&](auto& dst) {
                    qualified = table_piece(*table, src, dst, piece, ld.patches[piece.dst].box);
                });
            });
            if (!qualified)
            {
                table->usable = false;
                break;
            }
        }
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

    // the sums' groups, in fold_sums' order: pieces by destination and resources, then image
    template<std::size_t dim, typename PatchData>
    void build_sums(Plan<dim> const& plan, DeviceTable<dim>& table,
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
            SumGroup<dim> g{table.pieces[order[begin]].dst, dst_box, first.box, table.sum_points,
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
            table.sum_points += g.bounds.size();
            table.groups.push_back(g);
            begin = end;
        }
        table.sums_built = true;
    }

    // a resource pair whose type gives the device append hooks (found by ADL)
    template<typename Pieces>
    concept DeviceAppendable = requires(Pieces const& v) {
        { can_append_items_on_device(v) } -> std::same_as<bool>;
        append_items_on_device(v);
    };

    // a piece of the pair's resource types: into the pair's pieces
    template<typename D, typename S, std::size_t dim>
    bool append_piece(std::vector<AppendPiece<D, S, dim>>& pieces, S const& src, D& dst,
                      auto const& piece)
    {
        pieces.push_back({dst, src, piece.box, piece.src.shift, piece.dst});
        return true;
    }

    // a piece of other types: the pair is not one resource pair, the host path
    template<typename D, typename S, std::size_t dim>
    bool append_piece(std::vector<AppendPiece<D, S, dim>>&, auto const&, auto&, auto const&)
    {
        return false;
    }

    // a pair's run: its resource type's device hook on its pieces
    struct AppendRun
    {
        void (*run)(void*);
        void* pieces;
    };

    // per thread: the pairs' piece slots in use by append_on_device calls, a nested call (from a
    //  hook) takes the slots after its caller's, whose pieces are still to run
    struct AppendSlots
    {
        explicit AppendSlots(std::size_t const n)
            : base{used()}
        {
            used() += n;
        }
        AppendSlots(AppendSlots const&)            = delete;
        AppendSlots& operator=(AppendSlots const&) = delete;
        ~AppendSlots() { used() = base; }

        static std::size_t& used()
        {
            thread_local std::size_t n = 0;
            return n;
        }

        std::size_t const base;
    };

    // per thread and resource pair type, a slot's pieces, empty: kept between copies, and until
    //  every pair of a copy is known to run (a deque: a slot stays put as later ones are added)
    template<typename Pieces>
    Pieces& append_pieces(std::size_t const slot)
    {
        thread_local std::deque<Pieces> kept;
        while (kept.size() <= slot)
            kept.emplace_back();
        auto& pieces = kept[slot];
        pieces.clear();
        return pieces;
    }

    // the pair's pieces (for_each_piece(add): add(src, dst, piece) for each) handed at once to its
    //  resource type's device hooks: a run, if every piece is of the pair's types and they can
    template<std::size_t dim, typename S, typename D>
        requires DeviceAppendable<std::vector<AppendPiece<D, S, dim>>>
    bool append_run(S const&, D&, std::size_t const slot, AppendRun& run, auto&& for_each_piece)
    {
        using Pieces = std::vector<AppendPiece<D, S, dim>>;
        auto& pieces = append_pieces<Pieces>(slot);
        bool same    = true;
        for_each_piece([&](auto const& src, auto& dst, auto const& piece) {
            same = append_piece(pieces, src, dst, piece) and same;
        });
        if (!same or !can_append_items_on_device(pieces))
            return false;
        run = {[](void* const p) {
                   auto& ran = *static_cast<Pieces*>(p);
                   append_items_on_device(ran);
                   ran.clear(); // the handles not held past the copy
               },
               &pieces};
        return true;
    }

    // a resource type without device append hooks: the host path
    template<std::size_t dim, typename S, typename D>
    bool append_run(S const&, D&, std::size_t, AppendRun&, auto&&)
    {
        return false;
    }

} // namespace detail

// Assign on the device: false (nothing done) when the plan's pieces do not all qualify
template<std::size_t dim, typename Access, typename PatchData>
bool assign_on_device(Plan<dim>& plan, Access const& access, LevelData<dim, PatchData>& ld)
{
    auto& table = detail::device_table(plan, access, ld);
    if (!table.usable)
        return false;
    if (table.total == 0)
        return true;
    auto const* const pieces = table.pieces.data();
    std::size_t const n      = table.pieces.size();
    kernel::launch(table.total, [=] DAMN_ALL() {
        auto const i = kernel::idx();
        // the last piece whose first point is <= i
        std::size_t lo = 0, hi = n;
        while (hi - lo > 1)
            if (auto const mid = (lo + hi) / 2; pieces[mid].first <= i)
                lo = mid;
            else
                hi = mid;
        auto const& pc = pieces[lo];
        auto const p   = detail::point_in(pc.box, i - pc.first);
        core::Point<int, dim> q;
        for (std::size_t d = 0; d < dim; ++d)
            q[d] = p[d] - pc.shift[d];
        pc.dst[detail::flat_in(pc.dst_box, p)] = pc.src[detail::flat_in(pc.src_box, q)];
    });
    return true;
}

// Sum on the device: false (nothing done) when the plan's pieces do not all qualify
template<std::size_t dim, typename Access, typename PatchData>
bool sum_on_device(Plan<dim>& plan, Access const& access, LevelData<dim, PatchData>& ld)
{
    auto& table = detail::device_table(plan, access, ld);
    if (!table.usable)
        return false;
    if (table.total == 0)
        return true;
    if (!table.sums_built)
        detail::build_sums(plan, table, ld);

    auto const* const pieces = table.pieces.data();
    std::size_t const n      = table.pieces.size();
    auto* const snapshot     = table.snapshot.data();
    kernel::launch(table.total, [=] DAMN_ALL() {
        auto const i   = kernel::idx();
        std::size_t lo = 0, hi = n;
        while (hi - lo > 1)
            if (auto const mid = (lo + hi) / 2; pieces[mid].first <= i)
                lo = mid;
            else
                hi = mid;
        auto const& pc = pieces[lo];
        auto const p   = detail::point_in(pc.box, i - pc.first);
        core::Point<int, dim> q;
        for (std::size_t d = 0; d < dim; ++d)
            q[d] = p[d] - pc.shift[d];
        snapshot[i] = pc.src[detail::flat_in(pc.src_box, q)];
    });

    auto const* const groups       = table.groups.data();
    std::size_t const ng           = table.groups.size();
    auto const* const contributors = table.contributors.data();
    kernel::launch(table.sum_points, [=] DAMN_ALL() {
        auto const i   = kernel::idx();
        std::size_t lo = 0, hi = ng;
        while (hi - lo > 1)
            if (auto const mid = (lo + hi) / 2; groups[mid].first <= i)
                lo = mid;
            else
                hi = mid;
        auto const& g = groups[lo];
        auto const p  = detail::point_in(g.bounds, i - g.first);
        bool summed   = false;
        for (auto c = g.begin; c < g.end; ++c)
            if (contributors[c] >= 0 and detail::contains(pieces[contributors[c]].box, p))
                summed = true;
        if (!summed)
            return;
        auto& d          = g.dst[detail::flat_in(g.dst_box, p)];
        double const own = d;
        double acc       = 0;
        for (auto c = g.begin; c < g.end; ++c)
            if (contributors[c] < 0)
                acc += own;
            else if (auto const& pc = pieces[contributors[c]]; detail::contains(pc.box, p))
                acc += snapshot[pc.first + detail::flat_in(pc.box, p)];
        d = acc;
    });
    return true;
}

// Append on the device: false (nothing done) when a piece is not local, or a pair's resources
//  have no device hooks or refuse (can_append_items_on_device)
template<std::size_t dim, typename Access, typename PatchData>
bool append_on_device(Plan<dim> const& plan, Access const& access, LevelData<dim, PatchData>& ld)
{
    core::ScratchVector<std::uint32_t> pairs;
    for (auto const& piece : plan.pieces)
    {
        if (!detail::is_local(piece))
            return false;
        if (std::find(pairs->begin(), pairs->end(), piece.pair) == pairs->end())
            pairs->push_back(piece.pair);
    }
    if (pairs->empty())
        return false; // nothing to do: the host path is as cheap

    detail::AppendSlots const slots{pairs->size()};
    core::ScratchVector<detail::AppendRun> runs;
    runs->resize(pairs->size());
    for (std::size_t k = 0; k < pairs->size(); ++k)
    {
        auto const pair   = (*pairs)[k];
        auto const& first = *std::find_if(plan.pieces.begin(), plan.pieces.end(),
                                          [&](auto const& p) { return p.pair == pair; });
        bool ok           = false;
        detail::src_of<dim, true>(access, ld.data[first.src.local], first, [&](auto const& src0) {
            detail::dst_of<dim, true>(access, ld.data[first.dst], first, [&](auto& dst0) {
                ok = detail::append_run<dim>(
                    src0, dst0, slots.base + k, (*runs)[k], [&](auto&& add) {
                        for (auto const& piece : plan.pieces)
                            if (piece.pair == pair)
                                detail::src_of<dim, true>(
                                    access, ld.data[piece.src.local], piece, [&](auto const& src) {
                                        detail::dst_of<dim, true>(
                                            access, ld.data[piece.dst], piece,
                                            [&](auto& dst) { add(src, dst, piece); });
                                    });
                    });
            });
        });
        if (!ok)
            return false;
    }
    for (auto const& run : *runs)
        run.run(run.pieces);
    return true;
}

} // namespace damn::amr

#endif // DAMN_HAVE_GPU

#endif /* DAMN_AMR_DEVICE_COPY_HPP */
