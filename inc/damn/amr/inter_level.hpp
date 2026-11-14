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
//       grow(coarsen(box), op.stencil) are gathered from the coarse patches' domains, then
//       BoxData:      op(fine_resource, fine_patch_box, box, BoxValues<T const> coarse)
//                     fills the fine points of box from the coarse values
//       ParticleData: op(fine_resource, fine_patch_box, box, std::span<Item const> coarse,
//                        shift)
//                     per coarse patch image: appends the fine items (e.g. split coarse ones)
//                     whose cell is in box; coarse items' cells are at cell(item) + shift in
//                     the coarse index space
//     with a pair's coarse_new set, coarse values are (1 - alpha) coarse + alpha coarse_new
//
//   coarsen(op, map, pairs, fine_ld, coarse_ld)
//     BoxData only: for each fine patch and coarse patch image under it,
//       op(fine_resource const&, fine_patch_box, BoxValues<T> out) fills out.box: the coarse
//       domain points under the fine patch's domain (fine ghosts are expected filled)
//     then they are assigned to the coarse patch
//
// sources are never modified. Pieces of other ranks are damn/mpi/inter_level.hpp's: the
// functions here throw for them. What goes where is planned on first use for a region,
// resources and stencil, and kept in the LevelMap
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
#include "damn/amr/particle_data.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/amr/neighbours.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <iterator>

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

    NO_DISCARD std::size_t index(core::Point<int, dim> const& p) const
    {
        std::size_t i = 0, stride = 1;
        for (std::size_t d = 0; d < dim; ++d)
        {
            i += static_cast<std::size_t>(p[d] - box.lower[d]) * stride;
            stride *= static_cast<std::size_t>(box.upper[d] - box.lower[d] + 1);
        }
        return i;
    }

    NO_DISCARD T& operator()(core::Point<int, dim> const& p) const { return data[index(p)]; }
};


namespace detail
{
    // fn(resource) for the first patch data of either level holding name, of the kind wanted
    template<std::size_t dim, bool particles, typename CoarseLD, typename FineLD, typename Fn>
    void with_any_resource(CoarseLD& coarse_ld, FineLD& fine_ld, std::string const& name, Fn&& fn);

    // the fine target boxes of a fine patch for a region
    template<std::size_t dim>
    NO_DISCARD std::vector<core::Box<int, dim>>
    fine_boxes(LevelRegion region, LevelMap<dim> const& map, Patch<dim> const& f, auto const& res);

    template<std::size_t dim>
    NO_DISCARD std::size_t aligned(std::size_t bytes);

    template<std::size_t dim, typename CoarseLD, typename FineLD>
    NO_DISCARD std::unique_ptr<LevelPlan<dim>>
    level_plan(LevelRegion region, int stencil, LevelMap<dim> const& map, CoarseLD& coarse_ld,
               FineLD& fine_ld, std::vector<LevelPair> const& pairs);

    template<std::size_t dim>
    NO_DISCARD bool coarse_is_local(LevelPiece<dim> const& piece);

    template<std::size_t dim>
    NO_DISCARD bool fine_is_local(LevelPlan<dim> const& plan, LevelPiece<dim> const& piece);

    template<std::size_t dim>
    void check_local(LevelPlan<dim> const& plan);
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

    typename LevelMap<dim>::PlanKey key{region, pairs, stencil};
    auto found = map.plans.find(key);
    if (found == map.plans.end())
    {
        DAMN_LOG_SCOPE(2, "damn::amr::level_plan_for"); // once per region and resources
        found = map.plans
                    .emplace(std::move(key),
                             detail::level_plan(region, stencil, map, coarse_ld, fine_ld, pairs))
                    .first;
    }
    return *found->second;
}


// ---------------------------------------------------------------- pieces as bytes
// shared by the serial functions below and damn::mpi's

// refine: a piece's coarse data from its (local) coarse patch, appended to out: values in the
// box's point order, or the number of items then the items
template<std::size_t dim, typename CoarseLD>
void pack_coarse_piece(LevelPlan<dim> const& plan, LevelPiece<dim> const& piece,
                       CoarseLD& coarse_ld, double const alpha, std::vector<std::byte>& out)
{
    auto const& pair  = plan.pairs[plan.targets[piece.target].pair];
    auto& coarse_data = coarse_ld.data[piece.coarse.local];
    auto const& cbox  = piece.coarse.patch.box;
    auto const shift  = piece.coarse.shift;

    bool found = with_resource(coarse_data, pair.coarse, [&](auto const& src) {
        using R = std::decay_t<decltype(src)>;
        if constexpr (ParticleData<R, dim>)
        {
            auto const count_at = out.size();
            detail::write(out, std::uint64_t{0});
            std::uint64_t count = 0;
            for_each_item(src, [&](auto const& item) {
                if (core::isIn(cell(item) + shift, piece.box))
                {
                    detail::write(out, item);
                    ++count;
                }
            });
            std::memcpy(out.data() + count_at, &count, sizeof(count));
        }
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
            bool const found_new = with_resource(coarse_data, pair.coarse_new, [&](auto& nsrc) {
                if constexpr (std::is_same_v<std::decay_t<decltype(nsrc)>, R>)
                    core::for_each_point(piece.box, [&](auto const& p) {
                        detail::write(out, T((1 - alpha) * value(src, cbox, p - shift)
                                             + alpha * value(nsrc, cbox, p - shift)));
                    });
                else
                    throw std::runtime_error("damn::amr::refine: " + pair.coarse_new + " is not of "
                                             + pair.coarse + "'s kind");
            });
            if (!found_new)
                throw std::runtime_error("damn::amr::refine: no resource " + pair.coarse_new);
        }
        else
            throw std::runtime_error("damn::amr::refine: " + pair.coarse
                                     + " is neither BoxData nor ParticleData");
    });
    if (!found)
        throw std::runtime_error("damn::amr::refine: no resource " + pair.coarse);
}

// refine: a piece's coarse data, packed by pack_coarse_piece, into its (local) fine target:
// field values into the plan's scratch, items to op; at moves past it
template<std::size_t dim, typename FineLD>
void unpack_coarse_piece(LevelPlan<dim>& plan, LevelPiece<dim> const& piece, FineLD& fine_ld,
                         auto& op, std::byte const*& at)
{
    auto const& target = plan.targets[piece.target];
    auto const& pair   = plan.pairs[target.pair];
    auto& fine_data    = fine_ld.data[target.fine_local];

    bool found = with_resource(fine_data, pair.fine, [&](auto& dst) {
        using R = std::decay_t<decltype(dst)>;
        if constexpr (ParticleData<R, dim>)
        {
            using Item       = typename R::value_type;
            auto const count = detail::read<std::uint64_t>(at);
            std::vector<Item> items(count);
            if (count)
                std::memcpy(items.data(), at, count * sizeof(Item));
            at += count * sizeof(Item);
            op(dst, target.fine.box, target.box, std::span<Item const>{items}, piece.coarse.shift);
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
template<std::size_t dim, typename FineLD>
void clear_scratch(LevelPlan<dim>& plan, FineLD& fine_ld)
{
    for (auto const& target : plan.targets)
    {
        if (target.fine_local == not_local)
            continue;
        with_resource(
            fine_ld.data[target.fine_local], plan.pairs[target.pair].fine, [&](auto& dst) {
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
template<std::size_t dim, typename FineLD>
void refine_fields(LevelPlan<dim>& plan, FineLD& fine_ld, auto& op)
{
    for (auto const& target : plan.targets)
    {
        if (target.fine_local == not_local)
            continue;
        auto const& pair = plan.pairs[target.pair];
        with_resource(fine_ld.data[target.fine_local], pair.fine, [&](auto& dst) {
            using R = std::decay_t<decltype(dst)>;
            if constexpr (BoxData<R, dim>)
            {
                using T = box_data_value_t<R, dim>;
                BoxValues<T const, dim> coarse{
                    target.coarse_box,
                    reinterpret_cast<T const*>(
                        reinterpret_cast<std::byte const*>(plan.scratch.data()) + target.offset)};
                op(dst, target.fine.box, target.box, coarse);
            }
        });
    }
}

// coarsen: a target's coarse values computed by op from its (local) fine patch, appended to out
template<std::size_t dim, typename FineLD>
void pack_coarsened(LevelPlan<dim> const& plan, LevelTarget<dim> const& target, FineLD& fine_ld,
                    auto& op, std::vector<std::byte>& out)
{
    auto const& pair = plan.pairs[target.pair];
    bool found = with_resource(fine_ld.data[target.fine_local], pair.fine, [&](auto const& src) {
        using R = std::decay_t<decltype(src)>;
        if constexpr (BoxData<R, dim>)
        {
            using T       = box_data_value_t<R, dim>;
            auto const at = out.size();
            out.resize(at + target.coarse_box.size() * sizeof(T));
            std::vector<T> values(target.coarse_box.size());
            op(src, target.fine.box, BoxValues<T, dim>{target.coarse_box, values.data()});
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
template<std::size_t dim, typename CoarseLD>
void unpack_coarsened(LevelPlan<dim> const& plan, LevelPiece<dim> const& piece, CoarseLD& coarse_ld,
                      std::byte const*& at)
{
    auto const& pair = plan.pairs[plan.targets[piece.target].pair];
    auto const& cbox = piece.coarse.patch.box;
    bool found = with_resource(coarse_ld.data[piece.coarse.local], pair.coarse, [&](auto& dst) {
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

template<std::size_t dim, typename CoarseLD, typename FineLD>
void refine(auto op, LevelRegion const region, LevelMap<dim>& map,
            std::vector<LevelPair> const& pairs, CoarseLD& coarse_ld, FineLD& fine_ld,
            double const alpha = 0)
{
    DAMN_LOG_SCOPE(2, "damn::amr::refine");
    if (region == LevelRegion::Coarsened)
        throw std::runtime_error("damn::amr::refine: not a refine region");
    auto& plan = level_plan_for(region, op.stencil, map, pairs, coarse_ld, fine_ld);
    detail::check_local(plan);
    clear_scratch(plan, fine_ld);

    std::vector<std::byte> bytes;
    for (auto const& piece : plan.pieces)
    {
        bytes.clear();
        pack_coarse_piece(plan, piece, coarse_ld, alpha, bytes);
        std::byte const* at = bytes.data();
        unpack_coarse_piece(plan, piece, fine_ld, op, at);
    }
    refine_fields(plan, fine_ld, op);
}

template<std::size_t dim, typename FineLD, typename CoarseLD>
void coarsen(auto op, LevelMap<dim>& map, std::vector<LevelPair> const& pairs, FineLD& fine_ld,
             CoarseLD& coarse_ld)
{
    DAMN_LOG_SCOPE(2, "damn::amr::coarsen");
    auto& plan = level_plan_for(LevelRegion::Coarsened, 0, map, pairs, coarse_ld, fine_ld);
    detail::check_local(plan);

    std::vector<std::byte> bytes;
    for (auto const& piece : plan.pieces) // one per target
    {
        bytes.clear();
        pack_coarsened(plan, plan.targets[piece.target], fine_ld, op, bytes);
        std::byte const* at = bytes.data();
        unpack_coarsened(plan, piece, coarse_ld, at);
    }
}


// ---------------------------------------------------------------- detail

template<std::size_t dim, bool particles, typename CoarseLD, typename FineLD, typename Fn>
void detail::with_any_resource(CoarseLD& coarse_ld, FineLD& fine_ld, std::string const& name,
                               Fn&& fn)
{
    auto const try_in = [&](auto& ld) {
        if (ld.data.empty())
            return false;
        return with_resource(ld.data[0], name, [&](auto const& r) {
            using R = std::decay_t<decltype(r)>;
            if constexpr (particles)
            {
                if constexpr (ParticleData<R, dim>)
                    fn(r);
                else
                    throw std::runtime_error("damn::amr::inter_level: " + name
                                             + " is not ParticleData");
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
        bool const particles = [&] {
            bool is          = false;
            auto const probe = [&](auto& ld) {
                return !ld.data.empty()
                       and with_resource(ld.data[0], pair.fine, [&](auto const& r) {
                               is = ParticleData<std::decay_t<decltype(r)>, dim>;
                           });
            };
            if (!probe(fine_ld) and !probe(coarse_ld))
                throw std::runtime_error("damn::amr::inter_level: no resource " + pair.fine);
            return is;
        }();
        if (particles and region == LevelRegion::Coarsened)
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
                    if (!particles and fine_local != not_local)
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
                }
            }
        };

        if (particles)
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
    return plan;
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

} // namespace damn::amr

#endif // DAMN_AMR_INTER_LEVEL_HPP
