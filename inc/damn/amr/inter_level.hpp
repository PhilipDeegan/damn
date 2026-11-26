#ifndef DAMN_AMR_INTER_LEVEL_HPP
#define DAMN_AMR_INTER_LEVEL_HPP

// moving data between two consecutive levels, along a LevelMap: damn finds what goes where and
// moves it, the user's operators do the arithmetic (they know the centerings and the physics)
//
//   refine(op, region, map, pairs, coarse_ld, fine_ld, alpha = 0)
//     region Interior:    the fine patches' domain points (or cells), e.g. a new level
//     region LevelGhosts: fine ghost points (or cells) on no fine patch's domain, periodic
//                         images included: the coarse-fine border
//     for each fine target box, the coarse points (or cells) of
//       grow(coarsen(box), op.stencil) are gathered from the coarse patches' domains; field
//       points on no coarse domain (past the coarse level's border: a fine level nearer the
//       coarse level's border than its ghosts' reach, nesting_buffer 1 say) from the first
//       coarse patch, in level order, whose ghosts hold them (the coarse level's level ghosts,
//       filled from the level below it, so a level only ever reads the next coarser one), then
//       BoxData:      op(fine_resource, fine_patch_box, box, BoxValues<T const> coarse)
//                     fills the fine points of box from the coarse values
//       CellItemData: op(fine_resource, fine_patch_box, box, std::span<Item const> coarse,
//                        shift)
//                     per coarse patch image: appends the fine items (e.g. split coarse ones)
//                     whose cell is in box; coarse items' cells are at cell(item) + shift in
//                     the coarse index space
//     with a pair's coarse_new set, coarse values are (1 - alpha) coarse + alpha coarse_new:
//     coarse only at alpha 0, coarse_new only at alpha 1 (the other may be unset)
//
//   coarsen(op, map, pairs, fine_ld, coarse_ld)
//     BoxData only: for each fine patch and coarse patch image under it,
//       op(fine_resource const&, fine_patch_box, BoxValues<T> out) fills out.box: the coarse
//       domain points under the fine patch's domain (fine ghosts are expected filled)
//     then they are assigned to the coarse patch
//
// access: how the transfers reach a pair's resources on a patch's data, as damn::amr::copy's
//  (copy.hpp): by_name (the default), or the caller's accessors, one per pair in the pairs'
//  order, access[i].coarse(data), .coarse_new(data) (pairs interpolated in time) and
//  .fine(data) returning the typed resources (a reference, or a handle holding a pointer).
//  Plans are still made from the names
//
// an operator handles the resource kinds it has a call for: one asked to move a resource it
// has none for throws. Sources are never modified. Pieces of other ranks are
// damn/mpi/inter_level.hpp's: the functions here throw for them. What goes where is planned on
// first use for a region, resources and stencil, and kept in the LevelMap
//
// resources' boxes come from the user's domain_box / ghost_box of the resource kind, from any
// patch data holding the name (either level): the coarse and fine resources of a name are
// expected of the same kind

#include "damn/core/def.hpp"
#include "damn/core/logger.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/amr/box_data.hpp"
#include "damn/amr/copy.hpp"
#include "damn/amr/cell_item_data.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/cell_items.hpp"
#include "damn/core/memory/scratch.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>
#include <utility>

namespace damn::amr
{
template<std::size_t dim>
struct Patch;

// values on the points of a box, flat (x fastest), not owned
template<typename T, std::size_t dim>
struct BoxValues
{
    core::Box<int, dim> box{};
    T* data = nullptr;

    MKN_KUL_FN_SIG_RET std::size_t index(core::Point<int, dim> const& p) const
    {
        std::size_t i = 0, stride = 1;
        for (std::size_t d = 0; d < dim; ++d)
        {
            i += static_cast<std::size_t>(p[d] - box.lower[d]) * stride;
            stride *= static_cast<std::size_t>(box.upper[d] - box.lower[d] + 1);
        }
        return i;
    }

    MKN_KUL_FN_SIG_RET T& operator()(core::Point<int, dim> const& p) const
    {
        return data[index(p)];
    }
};


namespace detail
{
    // (1 - alpha) a + alpha b: arithmetic values, or arrays of them element by element
    template<typename T>
    MKN_KUL_FN_SIG_RET T interpolate(T const& a, T const& b, double alpha);
    template<typename T, std::size_t N>
    MKN_KUL_FN_SIG_RET std::array<T, N> interpolate(std::array<T, N> const& a,
                                                    std::array<T, N> const& b, double alpha);

    // a pair's side: the coarse resource, the coarse one at the step's end, the fine one
    enum class Side : std::uint8_t { coarse, coarse_new, fine };

    // fn(pair i's resource of side on data), as access reaches it: false if none (by name)
    template<Side side, typename Data, typename Fn>
    bool with_side(ByName, Data& data, LevelPair const& pair, std::size_t i, Fn&& fn);
    template<Side side, typename Access, typename Data, typename Fn>
        requires(!std::is_same_v<Access, ByName>)
    bool with_side(Access const& access, Data& data, LevelPair const& pair, std::size_t i, Fn&& fn);

    // fn(resource) for the first patch data of either level holding name, of the kind wanted
    template<std::size_t dim, bool items, typename CoarseLD, typename FineLD, typename Fn>
    void with_any_resource(CoarseLD& coarse_ld, FineLD& fine_ld, std::string const& name, Fn&& fn);

    // the fine target boxes of a fine patch for a region
    template<std::size_t dim>
    MKN_KUL_FN_SIG_RET std::vector<core::Box<int, dim>>
    fine_boxes(LevelRegion region, LevelMap<dim> const& map, Patch<dim> const& f, auto const& res);

    // refine: coarse points past the coarse level's domains, from coarse patches' ghosts
    template<std::size_t dim>
    MKN_KUL_FN_SIG_RET std::vector<std::pair<Link<dim>, core::Box<int, dim>>>
    coarse_ghost_sources(LevelMap<dim> const& map, Patch<dim> const& f, auto const& cres,
                         core::Box<int, dim> const& coarse_box,
                         std::vector<Link<dim>> const& links);

    template<std::size_t dim>
    MKN_KUL_FN_SIG_RET std::size_t aligned(std::size_t bytes);

    template<std::size_t dim, typename CoarseLD, typename FineLD>
    MKN_KUL_FN_SIG_RET std::unique_ptr<LevelPlan<dim>>
    level_plan(LevelRegion region, int stencil, LevelMap<dim> const& map, CoarseLD& coarse_ld,
               FineLD& fine_ld, std::vector<LevelPair> const& pairs);

    template<std::size_t dim>
    MKN_KUL_FN_SIG_RET bool coarse_is_local(LevelPiece<dim> const& piece);

    template<std::size_t dim>
    MKN_KUL_FN_SIG_RET bool fine_is_local(LevelPlan<dim> const& plan, LevelPiece<dim> const& piece);

    template<std::size_t dim>
    void check_local(LevelPlan<dim> const& plan);

    // plan.from: refine, my fine targets' pieces of other ranks' coarse patches; coarsen, my
    //  coarse patches' pieces of other ranks' fine patches
    template<std::size_t dim>
    void plan_from(LevelPlan<dim>& plan);

} // namespace detail


// the plan of a region for resource pairs and a stencil, made on first use
template<std::size_t dim, typename CoarseLD, typename FineLD>
LevelPlan<dim>& level_plan_for(LevelRegion const region, int const stencil, LevelMap<dim>& map,
                               std::vector<LevelPair> const& pairs, CoarseLD& coarse_ld,
                               FineLD& fine_ld)
{
    if (coarse_ld.level != map.coarse or coarse_ld.patches != map.local_coarse
        or fine_ld.level != map.fine or fine_ld.patches != map.local_fine)
        throw std::runtime_error("damn::amr::inter_level: the level map is not the level data's"
                                 " (rebuild it after a regrid)");

    auto found = map.plans.find(std::forward_as_tuple(region, pairs, stencil));
    if (found == map.plans.end())
    {
        DAMN_LOG_SCOPE(2, "damn::amr::level_plan_for"); // once per region and resources
        found = map.plans
                    .emplace(typename LevelMap<dim>::PlanKey{region, pairs, stencil},
                             detail::level_plan(region, stencil, map, coarse_ld, fine_ld, pairs))
                    .first;
    }
    return *found->second;
}


// ---------------------------------------------------------------- pieces as bytes
// shared by the serial functions below and damn::mpi's

// refine: a piece's coarse data from its (local) coarse patch, appended to out: values in the
// box's point order, or the number of items then the items
template<std::size_t dim, typename Access, typename CoarseLD>
void pack_coarse_piece(LevelPlan<dim> const& plan, LevelPiece<dim> const& piece,
                       Access const& access, CoarseLD& coarse_ld, double const alpha,
                       std::vector<std::byte>& out)
{
    using detail::Side;
    auto const pi     = plan.targets[piece.target].pair;
    auto const& pair  = plan.pairs[pi];
    auto& coarse_data = coarse_ld.data[piece.coarse.local];
    auto const& cbox  = piece.coarse.patch.box;
    auto const shift  = piece.coarse.shift;

    bool found
        = detail::with_side<Side::coarse>(access, coarse_data, pair, pi, [&](auto const& src) {
              using R = std::decay_t<decltype(src)>;
              if constexpr (CellItemData<R, dim>)
                  cell_items::pack(src, piece.box, shift, out);
              else if constexpr (BoxData<R, dim>)
              {
                  using T = box_data_value_t<R, dim>;
                  out.reserve(out.size() + piece.bytes);
                  if (pair.coarse_new.empty() or alpha == 0)
                  {
                      core::for_each_point(piece.box, [&](auto const& p) {
                          detail::write(out, T{value(src, cbox, p - shift)});
                      });
                      return;
                  }
                  // at alpha 1 coarse_new only: coarse may not be set yet (0 * NaN is NaN)
                  bool const found_new = detail::with_side<Side::coarse_new>(
                      access, coarse_data, pair, pi, [&](auto& nsrc) {
                          if constexpr (std::is_same_v<std::decay_t<decltype(nsrc)>, R>)
                          {
                              if (alpha == 1)
                                  core::for_each_point(piece.box, [&](auto const& p) {
                                      detail::write(out, T{value(nsrc, cbox, p - shift)});
                                  });
                              else
                                  core::for_each_point(piece.box, [&](auto const& p) {
                                      detail::write(
                                          out, detail::interpolate(T{value(src, cbox, p - shift)},
                                                                   T{value(nsrc, cbox, p - shift)},
                                                                   alpha));
                                  });
                          }
                          else
                              throw std::runtime_error("damn::amr::refine: " + pair.coarse_new
                                                       + " is not of " + pair.coarse + "'s kind");
                      });
                  if (!found_new)
                      throw std::runtime_error("damn::amr::refine: no resource " + pair.coarse_new);
              }
              else
                  throw std::runtime_error("damn::amr::refine: " + pair.coarse
                                           + " is neither BoxData nor CellItemData");
          });
    if (!found)
        throw std::runtime_error("damn::amr::refine: no resource " + pair.coarse);
}

// refine: a piece's coarse data, packed by pack_coarse_piece, into its (local) fine target:
// field values into the plan's scratch, items to op; at moves past it
template<std::size_t dim, typename Access, typename FineLD>
void unpack_coarse_piece(LevelPlan<dim>& plan, LevelPiece<dim> const& piece, Access const& access,
                         FineLD& fine_ld, auto& op, std::byte const*& at)
{
    auto const& target = plan.targets[piece.target];
    auto const& pair   = plan.pairs[target.pair];
    auto& fine_data    = fine_ld.data[target.fine_local];

    bool found = detail::with_side<
        detail::Side::fine>(access, fine_data, pair, target.pair, [&](auto& dst) {
        using R = std::decay_t<decltype(dst)>;
        if constexpr (CellItemData<R, dim>)
        {
            using Item       = typename R::value_type;
            auto const items = cell_items::read(dst, at); // the coarse items
            if constexpr (std::invocable<decltype(op)&, R&, core::Box<int, dim> const&,
                                         core::Box<int, dim> const&, std::span<Item const>,
                                         core::Point<int, dim> const&>)
                op(dst, target.fine.box, target.box, std::span<Item const>{items},
                   piece.coarse.shift);
            else
                throw std::runtime_error("damn::amr::refine: the operator does not refine "
                                         + pair.fine);
        }
        else if constexpr (BoxData<R, dim>)
        {
            using T = box_data_value_t<R, dim>;
            BoxValues<T, dim> coarse{
                target.coarse_box,
                reinterpret_cast<T*>(reinterpret_cast<std::byte*>(plan.scratch.data())
                                     + target.offset)};
            core::for_each_point(piece.box,
                                 [&](auto const& p) { coarse(p) = detail::read<T>(at); });
        }
    });
    if (!found)
        throw std::runtime_error("damn::amr::refine: no resource " + pair.fine);
}

// refine: local field targets' coarse values NaN (or value initialized), so fine points
// without coarse data under them (past a non periodic edge) are never stale
template<std::size_t dim, typename Access, typename FineLD>
void clear_scratch(LevelPlan<dim>& plan, Access const& access, FineLD& fine_ld)
{
    for (auto const& target : plan.targets)
    {
        if (target.fine_local == not_local)
            continue;
        detail::with_side<detail::Side::fine>(
            access, fine_ld.data[target.fine_local], plan.pairs[target.pair], target.pair,
            [&](auto& dst) {
                using R = std::decay_t<decltype(dst)>;
                if constexpr (BoxData<R, dim>)
                {
                    using T = box_data_value_t<R, dim>;
                    T fill{};
                    if constexpr (std::numeric_limits<T>::has_quiet_NaN)
                        fill = std::numeric_limits<T>::quiet_NaN();
                    auto* at = reinterpret_cast<T*>(
                        reinterpret_cast<std::byte*>(plan.scratch.data()) + target.offset);
                    std::fill(at, at + target.coarse_box.size(), fill);
                }
            });
    }
}

// refine: the fine points of every local field target from its gathered coarse values
template<std::size_t dim, typename Access, typename FineLD>
void refine_fields(LevelPlan<dim>& plan, Access const& access, FineLD& fine_ld, auto& op)
{
    for (auto const& target : plan.targets)
    {
        if (target.fine_local == not_local)
            continue;
        auto const& pair = plan.pairs[target.pair];
        detail::with_side<detail::Side::fine>(
            access, fine_ld.data[target.fine_local], pair, target.pair, [&](auto& dst) {
                using R = std::decay_t<decltype(dst)>;
                if constexpr (BoxData<R, dim>)
                {
                    using T = box_data_value_t<R, dim>;
                    BoxValues<T const, dim> coarse{
                        target.coarse_box,
                        reinterpret_cast<T const*>(
                            reinterpret_cast<std::byte const*>(plan.scratch.data())
                            + target.offset)};
                    if constexpr (std::invocable<decltype(op)&, R&, core::Box<int, dim> const&,
                                                 core::Box<int, dim> const&,
                                                 decltype(coarse) const&>)
                        op(dst, target.fine.box, target.box, coarse);
                    else
                        throw std::runtime_error("damn::amr::refine: the operator does not refine "
                                                 + pair.fine);
                }
            });
    }
}

// coarsen: a target's coarse values computed by op from its (local) fine patch, appended to out
template<std::size_t dim, typename Access, typename FineLD>
void pack_coarsened(LevelPlan<dim> const& plan, LevelTarget<dim> const& target,
                    Access const& access, FineLD& fine_ld, auto& op, std::vector<std::byte>& out)
{
    auto const& pair = plan.pairs[target.pair];
    bool found       = detail::with_side<detail::Side::fine>(
        access, fine_ld.data[target.fine_local], pair, target.pair, [&](auto const& src) {
            using R = std::decay_t<decltype(src)>;
            if constexpr (BoxData<R, dim>)
            {
                using T       = box_data_value_t<R, dim>;
                auto const at = out.size();
                out.resize(at + target.coarse_box.size() * sizeof(T));
                // value initialized as a vector's, whatever op leaves unwritten
                core::Scratch<T> values{static_cast<std::size_t>(target.coarse_box.size())};
                std::fill_n(values.data(), values.size(), T{});
                if constexpr (std::invocable<decltype(op)&, R const&, core::Box<int, dim> const&,
                                             BoxValues<T, dim> const&>)
                    op(src, target.fine.box, BoxValues<T, dim>{target.coarse_box, values.data()});
                else
                    throw std::runtime_error("damn::amr::coarsen: the operator does not coarsen "
                                             + pair.fine);
                // in the box's point order
                BoxValues<T const, dim> const computed{target.coarse_box, values.data()};
                auto* to = out.data() + at;
                core::for_each_point(target.coarse_box, [&](auto const& p) {
                    std::memcpy(to, &computed(p), sizeof(T));
                    to += sizeof(T);
                });
            }
            else
                throw std::runtime_error("damn::amr::coarsen: " + pair.fine + " is not BoxData");
        });
    if (!found)
        throw std::runtime_error("damn::amr::coarsen: no resource " + pair.fine);
}

// coarsen: a piece's coarse values, packed by pack_coarsened, into its (local) coarse patch
template<std::size_t dim, typename Access, typename CoarseLD>
void unpack_coarsened(LevelPlan<dim> const& plan, LevelPiece<dim> const& piece,
                      Access const& access, CoarseLD& coarse_ld, std::byte const*& at)
{
    auto const pi    = plan.targets[piece.target].pair;
    auto const& pair = plan.pairs[pi];
    auto const& cbox = piece.coarse.patch.box;
    bool found       = detail::with_side<detail::Side::coarse>(
        access, coarse_ld.data[piece.coarse.local], pair, pi, [&](auto& dst) {
            using R = std::decay_t<decltype(dst)>;
            if constexpr (BoxData<R, dim>)
            {
                using T = box_data_value_t<R, dim>;
                core::for_each_point(piece.box, [&](auto const& p) {
                    value(dst, cbox, p - piece.coarse.shift) = detail::read<T>(at);
                });
            }
        });
    if (!found)
        throw std::runtime_error("damn::amr::coarsen: no resource " + pair.coarse);
}


// ---------------------------------------------------------------- one rank

// level data: a LevelData (what an access argument is not)
template<typename T>
concept LevelDataLike = requires(T& t) {
    t.data;
    t.patches;
    t.level;
};

template<std::size_t dim, typename Access, typename CoarseLD, typename FineLD>
    requires LevelDataLike<FineLD>
void refine(auto op, LevelRegion const region, LevelMap<dim>& map,
            std::vector<LevelPair> const& pairs, Access const& access, CoarseLD& coarse_ld,
            FineLD& fine_ld, double const alpha = 0)
{
    DAMN_LOG_SCOPE(2, "damn::amr::refine");
    if (region == LevelRegion::Coarsened)
        throw std::runtime_error("damn::amr::refine: not a refine region");
    check_access(access, pairs.size());
    auto& plan = level_plan_for(region, op.stencil, map, pairs, coarse_ld, fine_ld);
    detail::check_local(plan);
    clear_scratch(plan, access, fine_ld);

    for (auto const& piece : plan.pieces)
    {
        plan.local.clear();
        pack_coarse_piece(plan, piece, access, coarse_ld, alpha, plan.local);
        std::byte const* at = plan.local.data();
        unpack_coarse_piece(plan, piece, access, fine_ld, op, at);
    }
    refine_fields(plan, access, fine_ld, op);
}

template<std::size_t dim, typename CoarseLD, typename FineLD>
    requires LevelDataLike<FineLD>
void refine(auto op, LevelRegion const region, LevelMap<dim>& map,
            std::vector<LevelPair> const& pairs, CoarseLD& coarse_ld, FineLD& fine_ld,
            double const alpha = 0)
{
    refine(op, region, map, pairs, by_name, coarse_ld, fine_ld, alpha);
}

template<std::size_t dim, typename Access, typename FineLD, typename CoarseLD>
    requires LevelDataLike<CoarseLD>
void coarsen(auto op, LevelMap<dim>& map, std::vector<LevelPair> const& pairs, Access const& access,
             FineLD& fine_ld, CoarseLD& coarse_ld)
{
    DAMN_LOG_SCOPE(2, "damn::amr::coarsen");
    check_access(access, pairs.size());
    auto& plan = level_plan_for(LevelRegion::Coarsened, 0, map, pairs, coarse_ld, fine_ld);
    detail::check_local(plan);

    for (auto const& piece : plan.pieces) // one per target
    {
        plan.local.clear();
        pack_coarsened(plan, plan.targets[piece.target], access, fine_ld, op, plan.local);
        std::byte const* at = plan.local.data();
        unpack_coarsened(plan, piece, access, coarse_ld, at);
    }
}

template<std::size_t dim, typename FineLD, typename CoarseLD>
    requires LevelDataLike<CoarseLD>
void coarsen(auto op, LevelMap<dim>& map, std::vector<LevelPair> const& pairs, FineLD& fine_ld,
             CoarseLD& coarse_ld)
{
    coarsen(op, map, pairs, by_name, fine_ld, coarse_ld);
}


// ---------------------------------------------------------------- detail

namespace detail
{
    template<Side side>
    MKN_KUL_FN_SIG_RET std::string const& side_name(LevelPair const& pair)
        requires(side == Side::coarse)
    {
        return pair.coarse;
    }
    template<Side side>
    MKN_KUL_FN_SIG_RET std::string const& side_name(LevelPair const& pair)
        requires(side == Side::coarse_new)
    {
        return pair.coarse_new;
    }
    template<Side side>
    MKN_KUL_FN_SIG_RET std::string const& side_name(LevelPair const& pair)
        requires(side == Side::fine)
    {
        return pair.fine;
    }

    template<Side side>
    MKN_KUL_FN_SIG_RET decltype(auto) side_of(auto const& accessor, auto& data)
        requires(side == Side::coarse)
    {
        return accessor.coarse(data);
    }
    template<Side side>
    MKN_KUL_FN_SIG_RET decltype(auto) side_of(auto const& accessor, auto& data)
        requires(side == Side::coarse_new)
    {
        return accessor.coarse_new(data);
    }
    template<Side side>
    MKN_KUL_FN_SIG_RET decltype(auto) side_of(auto const& accessor, auto& data)
        requires(side == Side::fine)
    {
        return accessor.fine(data);
    }
} // namespace detail

template<detail::Side side, typename Data, typename Fn>
bool detail::with_side(ByName, Data& data, LevelPair const& pair, std::size_t, Fn&& fn)
{
    return with_resource(data, side_name<side>(pair), fn);
}

template<detail::Side side, typename Access, typename Data, typename Fn>
    requires(!std::is_same_v<Access, ByName>)
bool detail::with_side(Access const& access, Data& data, LevelPair const&, std::size_t const i,
                       Fn&& fn)
{
    decltype(auto) resource = side_of<side>(access[i], data); // a reference or a handle
    fn(resource);
    return true;
}

template<std::size_t dim, bool items, typename CoarseLD, typename FineLD, typename Fn>
void detail::with_any_resource(CoarseLD& coarse_ld, FineLD& fine_ld, std::string const& name,
                               Fn&& fn)
{
    // a rank without patches of one level still plans the other's sends: that level's
    //  resources from its prototype (two patch data types: the names differ)
    auto const try_in = [&](auto& ld) {
        auto* const any = any_patch_data(ld);
        if (!any)
            return false;
        return with_resource(*any, name, [&](auto const& r) {
            using R = std::decay_t<decltype(r)>;
            if constexpr (items)
            {
                if constexpr (CellItemData<R, dim>)
                    fn(r);
                else
                    throw std::runtime_error("damn::amr::inter_level: " + name
                                             + " is not CellItemData");
            }
            else if constexpr (BoxData<R, dim>)
                fn(r);
            else
                throw std::runtime_error("damn::amr::inter_level: " + name + " is not BoxData");
        });
    };
    if (!try_in(fine_ld) and !try_in(coarse_ld))
        throw std::runtime_error("damn::amr::inter_level: no resource " + name);
}

template<std::size_t dim>
std::vector<core::Box<int, dim>> detail::fine_boxes(LevelRegion const region,
                                                    LevelMap<dim> const& map, Patch<dim> const& f,
                                                    auto const& res)
{
    if (region == LevelRegion::Interior)
        return {domain_box(res, f.box)};

    // LevelGhosts: the ghost points minus every fine patch's domain, periodic images included
    std::vector<core::Box<int, dim>> boxes{ghost_box(res, f.box)};
    for (auto const& g : map.fine_built_on)
        for (auto const& s : map.fine_shifts)
        {
            auto const domain = domain_box(res, core::shift(g.box, s));
            if (!(domain * ghost_box(res, f.box)))
                continue;
            std::vector<core::Box<int, dim>> rest;
            for (auto const& box : boxes)
                for (auto const& r : box.remove(domain))
                    rest.push_back(r);
            boxes = std::move(rest);
        }
    return boxes;
}

// refine: the parts of coarse_box on no coarse patch's domain (images included), each from the
// first coarse patch image, in level order, under f (as the links: its box meets f's footprint)
// whose ghost box holds it. Decided from the coarse level's geometry, the same on every rank;
// returned for the images in links only (another rank's fine patch: my coarse patches')
template<std::size_t dim>
std::vector<std::pair<Link<dim>, core::Box<int, dim>>>
detail::coarse_ghost_sources(LevelMap<dim> const& map, Patch<dim> const& f, auto const& cres,
                             core::Box<int, dim> const& coarse_box,
                             std::vector<Link<dim>> const& links)
{
    std::vector<core::Box<int, dim>> rest{coarse_box};
    auto const remove = [&](core::Box<int, dim> const& b) {
        std::vector<core::Box<int, dim>> left;
        for (auto const& r : rest)
            for (auto const& piece : r.remove(b))
                left.push_back(piece);
        rest = std::move(left);
    };
    for (auto const& c : map.coarse_built_on)
        for (auto const& s : map.coarse_shifts)
            if (!rest.empty())
                remove(domain_box(cres, core::shift(c.box, s)));
    if (rest.empty())
        return {};

    std::vector<std::pair<Link<dim>, core::Box<int, dim>>> sources;
    auto const footprint = core::coarsen(core::grow(f.box, map.reach), map.ratio);
    for (auto const& c : map.coarse_built_on)
        for (auto const& s : map.coarse_shifts)
        {
            if (rest.empty() or !(footprint * core::shift(c.box, s)))
                continue;
            auto const ghosts = ghost_box(cres, core::shift(c.box, s));
            std::vector<core::Box<int, dim>> held;
            for (auto const& r : rest)
                if (auto const overlap = r * ghosts)
                    held.push_back(*overlap);
            for (auto const& h : held)
            {
                remove(h);
                for (auto const& link : links)
                    if (link.patch.id == c.id and link.shift == s)
                        sources.emplace_back(link, h);
            }
        }
    return sources; // what is left: past a non periodic edge, NaN as before
}

template<std::size_t dim>
std::size_t detail::aligned(std::size_t const bytes)
{
    auto constexpr a = sizeof(std::max_align_t);
    return (bytes + a - 1) / a * a;
}

template<std::size_t dim, typename CoarseLD, typename FineLD>
std::unique_ptr<LevelPlan<dim>>
detail::level_plan(LevelRegion const region, int const stencil, LevelMap<dim> const& map,
                   CoarseLD& coarse_ld, FineLD& fine_ld, std::vector<LevelPair> const& pairs)
{
    auto plan           = std::make_unique<LevelPlan<dim>>();
    plan->region        = region;
    plan->pairs         = pairs;
    std::size_t scratch = 0; // bytes
    if (coarse_ld.data.empty() and fine_ld.data.empty())
        return plan; // nothing of either level here

    // fine patches with a side here, in level order, and their links (all of them for mine,
    //  to my coarse patches for other ranks')
    std::vector<std::pair<Patch<dim>, std::vector<Link<dim>> const*>> fines;
    {
        std::size_t r = 0, l = 0;
        for (auto const& f : map.fine_built_on)
            if (l < map.local_fine.size() and map.local_fine[l] == f)
                fines.emplace_back(f, &map.under[l++]);
            else if (r < map.remote.size() and map.remote[r].first == f)
                fines.emplace_back(f, &map.remote[r++].second);
    }

    for (std::size_t pi = 0; pi < pairs.size(); ++pi)
    {
        auto const& pair     = pairs[pi];
        bool const items     = [&] {
            bool is          = false;
            auto const probe = [&](auto& ld) {
                auto* const any = any_patch_data(ld);
                return any and with_resource(*any, pair.fine, [&](auto const& r) {
                           is = CellItemData<std::decay_t<decltype(r)>, dim>;
                       });
            };
            if (!probe(fine_ld) and !probe(coarse_ld))
                throw std::runtime_error("damn::amr::inter_level: no resource " + pair.fine);
            return is;
        }();
        if (items and region == LevelRegion::Coarsened)
            throw std::runtime_error("damn::amr::coarsen: " + pair.fine + " is not BoxData");

        auto const plan_pair = [&](auto const& fres, auto const& cres) {
            using CR               = std::decay_t<decltype(cres)>;
            std::size_t value_size = 0;
            if constexpr (BoxData<CR, dim>)
                value_size = sizeof(box_data_value_t<CR, dim>);

            for (std::size_t fi = 0; fi < fines.size(); ++fi)
            {
                auto const& [f, links] = fines[fi];
                std::size_t fine_local = not_local;
                for (std::size_t i = 0; i < map.local_fine.size(); ++i)
                    if (map.local_fine[i].id == f.id)
                        fine_local = i;

                // the coarse points reachable from f: what its links cover
                auto const reachable
                    = domain_box(cres, core::coarsen(core::grow(f.box, map.reach), map.ratio));

                if (region == LevelRegion::Coarsened)
                {
                    auto const under = domain_box(cres, core::coarsen(f.box, map.ratio));
                    for (auto const& link : *links)
                        if (auto const box
                            = under * domain_box(cres, core::shift(link.patch.box, link.shift)))
                        {
                            plan->targets.push_back({pi, f, fine_local, {}, *box, 0});
                            plan->pieces.push_back(
                                {plan->targets.size() - 1, link, *box, box->size() * value_size});
                        }
                    continue;
                }

                for (auto const& box : fine_boxes(region, map, f, fres))
                {
                    auto const coarse_box = core::grow(core::coarsen(box, map.ratio), stencil);
                    if (!core::isIn(coarse_box.lower, reachable)
                        or !core::isIn(coarse_box.upper, reachable))
                        throw std::runtime_error(
                            "damn::amr::refine: " + pair.fine
                            + " needs coarse data past the level map's reach of "
                            + std::to_string(map.reach));

                    std::size_t offset = 0;
                    if (!items and fine_local != not_local)
                    {
                        offset = scratch;
                        scratch += aligned<dim>(coarse_box.size() * value_size);
                    }
                    plan->targets.push_back({pi, f, fine_local, box, coarse_box, offset});
                    for (auto const& link : *links)
                    {
                        auto const src = domain_box(cres, core::shift(link.patch.box, link.shift));
                        if (auto const overlap = coarse_box * src)
                            plan->pieces.push_back({plan->targets.size() - 1, link, *overlap,
                                                    overlap->size() * value_size});
                    }
                    if (!items)
                        for (auto const& [link, overlap] :
                             coarse_ghost_sources(map, f, cres, coarse_box, *links))
                            plan->pieces.push_back({plan->targets.size() - 1, link, overlap,
                                                    overlap.size() * value_size});
                }
            }
        };

        if (items)
            with_any_resource<dim, true>(coarse_ld, fine_ld, pair.fine, [&](auto const& fres) {
                with_any_resource<dim, true>(coarse_ld, fine_ld, pair.coarse,
                                             [&](auto const& cres) { plan_pair(fres, cres); });
            });
        else
            with_any_resource<dim, false>(coarse_ld, fine_ld, pair.fine, [&](auto const& fres) {
                with_any_resource<dim, false>(coarse_ld, fine_ld, pair.coarse,
                                              [&](auto const& cres) { plan_pair(fres, cres); });
            });
    }

    plan->scratch.resize(scratch / sizeof(std::max_align_t));
    plan_from(*plan);
    return plan;
}

template<std::size_t dim>
void detail::plan_from(LevelPlan<dim>& plan)
{
    std::map<int, std::vector<std::size_t>> from;
    for (std::size_t i = 0; i < plan.pieces.size(); ++i)
    {
        auto const& piece = plan.pieces[i];
        if (plan.region == LevelRegion::Coarsened)
        {
            if (!fine_is_local(plan, piece) and coarse_is_local(piece))
                from[plan.targets[piece.target].fine.rank].push_back(i);
        }
        else if (fine_is_local(plan, piece) and !coarse_is_local(piece))
            from[piece.coarse.patch.rank].push_back(i);
    }
    plan.from.assign(from.begin(), from.end());
}

template<std::size_t dim>
bool detail::coarse_is_local(LevelPiece<dim> const& piece)
{
    return piece.coarse.local != not_local;
}

template<std::size_t dim>
bool detail::fine_is_local(LevelPlan<dim> const& plan, LevelPiece<dim> const& piece)
{
    return plan.targets[piece.target].fine_local != not_local;
}

template<std::size_t dim>
void detail::check_local(LevelPlan<dim> const& plan)
{
    for (auto const& piece : plan.pieces)
        if (!coarse_is_local(piece) or !fine_is_local(plan, piece))
            throw std::runtime_error(
                "damn::amr::inter_level: patches of other ranks, damn::mpi's refine/coarsen");
}

template<typename T>
T detail::interpolate(T const& a, T const& b, double const alpha)
{
    return T((1 - alpha) * a + alpha * b);
}

template<typename T, std::size_t N>
std::array<T, N> detail::interpolate(std::array<T, N> const& a, std::array<T, N> const& b,
                                     double const alpha)
{
    std::array<T, N> out;
    for (std::size_t i = 0; i < N; ++i)
        out[i] = interpolate(a[i], b[i], alpha);
    return out;
}

} // namespace damn::amr

#endif // DAMN_AMR_INTER_LEVEL_HPP
