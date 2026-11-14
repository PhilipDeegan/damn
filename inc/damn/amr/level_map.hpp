#ifndef DAMN_AMR_LEVEL_MAP_HPP
#define DAMN_AMR_LEVEL_MAP_HPP

// LevelMap: how the patches of two consecutive levels overlap, periodic images included
//
//   under[i]: for the i-th local fine patch, the coarse patches under its box grown by reach
//             (fine cells), e.g. to fill its level border ghosts or a new patch from coarse
//             data; link shifts are in the coarse index space
//   over[i]:  for the i-th local coarse patch, the fine patches over it, e.g. to coarsen fine
//             data onto it; link shifts are in the fine index space
//   remote:   other ranks' fine patches over my coarse patches, in level order, with their
//             links to my coarse patches only: what I send them (refine) or receive from them
//             (coarsen)
//
// geometric, as Neighbours: built per regrid of either level. damn/amr/inter_level.hpp refines
// and coarsens data along it, planning what goes where on first use and keeping the plans here

#include "damn/core/def.hpp"
#include "damn/core/logger.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/topology.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>
#include <iterator>

namespace damn::amr
{
// what moves between levels for a resource: coarse's resource and fine's (the same name for
// fields, usually); coarse_new: if set, refining reads (1 - alpha) coarse + alpha coarse_new,
// coarse data interpolated in time
struct LevelPair
{
    std::string coarse{};
    std::string fine{};
    std::string coarse_new{};

    auto operator<=>(LevelPair const&) const = default;
};

NO_DISCARD inline std::vector<LevelPair> same_level_names(std::vector<std::string> const& names)
{
    std::vector<LevelPair> pairs;
    pairs.reserve(names.size());
    for (auto const& name : names)
        pairs.push_back({name, name, {}});
    return pairs;
}

// time interpolated: names at the old and new coarse times
NO_DISCARD inline std::vector<LevelPair> same_level_names(std::vector<std::string> const& names,
                                                          std::string const& old_suffix,
                                                          std::string const& new_suffix)
{
    std::vector<LevelPair> pairs;
    pairs.reserve(names.size());
    for (auto const& name : names)
        pairs.push_back({name + old_suffix, name, name + new_suffix});
    return pairs;
}

// what goes where between levels, see inter_level.hpp
enum class LevelRegion : std::uint8_t {
    Interior,    // refine: the fine patches' domain points (or cells), e.g. a new level
    LevelGhosts, // refine: fine ghost points (or cells) on no fine patch's domain
    Coarsened,   // coarsen: the coarse domain points under the fine patches' domains
};

// one target of a transfer, on the fine side: fine points to fill from the coarse points
// around them (refine), or coarse points computed from a fine patch (coarsen)
template<std::size_t dim>
struct LevelTarget
{
    std::size_t pair = 0;               // index in the plan's pairs
    Patch<dim> fine{};                  // the fine patch
    std::size_t fine_local = not_local; // its LevelData index, if mine
    core::Box<int, dim> box{};          // refine: fine points (or cells) to fill
    core::Box<int, dim> coarse_box{};   // coarse points (or cells) read (refine) or computed
                                        //  (coarsen), fine patch's frame (no shift)
    std::size_t offset = 0;             // into the plan's scratch (bytes), fine local fields
};

// one coarse patch's part of a target: coarse points (or cells) in the fine patch's frame, the
// coarse patch's image at coarse.patch.box + coarse.shift
template<std::size_t dim>
struct LevelPiece
{
    std::size_t target = 0;
    Link<dim> coarse{};
    core::Box<int, dim> box{};
    std::size_t bytes = 0; // fields (0 for particles: their number varies)
};

// what goes where for a region, resources and stencil, from one rank's point of view: every
// target and piece with a side on this rank, in the same order on every rank (fine patches in
// level order, their links in level order), so messages need no description of their content
template<std::size_t dim>
struct LevelPlan
{
    LevelRegion region = LevelRegion::Interior;
    std::vector<LevelPair> pairs{};
    std::vector<LevelTarget<dim>> targets{};
    std::vector<LevelPiece<dim>> pieces{};
    std::vector<std::max_align_t> scratch{}; // coarse values of fine local field targets
    std::map<int, std::vector<std::byte>> sent{}, received{}; // per peer rank, kept
};

template<std::size_t dim>
struct LevelMap
{
    using PlanKey = std::tuple<LevelRegion, std::vector<LevelPair>, int>; // and the stencil

    int coarse = 0; // level numbers, fine == coarse + 1
    int fine   = 1;
    int rank   = 0;
    int reach  = 0; // fine cells
    int ratio  = 2;
    std::vector<Patch<dim>> coarse_built_on{}, fine_built_on{};
    std::vector<Patch<dim>> local_coarse{}, local_fine{}; // this rank's, in LevelData order
    std::vector<std::vector<Link<dim>>> under{};          // per local fine patch
    std::vector<std::vector<Link<dim>>> over{};           // per local coarse patch
    std::vector<std::pair<Patch<dim>, std::vector<Link<dim>>>> remote{};
    std::vector<core::Point<int, dim>> fine_shifts{}; // the fine level's periodic images, and
                                                      //  no shift first
    std::map<PlanKey, std::unique_ptr<LevelPlan<dim>>> plans{};
};


namespace detail
{
    // local patch id -> its LevelData index
    template<std::size_t dim>
    NO_DISCARD std::map<std::uint32_t, std::size_t>
    local_indices(std::vector<Patch<dim>> const& local);

    // the shifts with the unshifted image first
    template<std::size_t dim>
    NO_DISCARD std::vector<core::Point<int, dim>>
    with_no_shift(std::vector<core::Point<int, dim>> s);
} // namespace detail

template<std::size_t dim>
std::map<std::uint32_t, std::size_t> detail::local_indices(std::vector<Patch<dim>> const& local)
{
    std::map<std::uint32_t, std::size_t> index_of;
    for (std::size_t i = 0; i < local.size(); ++i)
        index_of[local[i].id] = i;
    return index_of;
}

template<std::size_t dim>
std::vector<core::Point<int, dim>> detail::with_no_shift(std::vector<core::Point<int, dim>> s)
{
    s.insert(s.begin(), core::Point<int, dim>{});
    return s;
}


// the map between level coarse and level coarse + 1, for rank's patches of both
template<std::size_t dim>
NO_DISCARD LevelMap<dim> make_level_map(Hierarchy<dim> const& h, int const coarse, int const rank,
                                        Topology const& topology, int const reach)
{
    DAMN_LOG_SCOPE(1, "damn::amr::make_level_map");
    int const fine = coarse + 1;
    if (!level_exists(h, coarse) or !level_exists(h, fine))
        throw std::runtime_error("damn::amr::make_level_map: no levels " + std::to_string(coarse)
                                 + " and " + std::to_string(fine));
    if (reach < 0)
        throw std::runtime_error("damn::amr::make_level_map: negative reach");

    auto const& cl = h.levels[coarse];
    auto const& fl = h.levels[fine];

    LevelMap<dim> map;
    map.coarse          = coarse;
    map.fine            = fine;
    map.rank            = rank;
    map.reach           = reach;
    map.ratio           = h.ratio;
    map.coarse_built_on = cl.patches;
    map.fine_built_on   = fl.patches;
    map.local_coarse    = local_patches(cl, rank);
    map.local_fine      = local_patches(fl, rank);

    auto const coarse_index = detail::local_indices(map.local_coarse);
    auto const fine_index   = detail::local_indices(map.local_fine);
    auto const local_of     = [](auto const& index, std::uint32_t const id) {
        auto const found = index.find(id);
        return found == index.end() ? not_local : found->second;
    };

    auto const coarse_shifts = detail::with_no_shift(periodic_shifts(h, coarse));
    auto const fine_shifts   = detail::with_no_shift(periodic_shifts(h, fine));

    // fine cells -> the coarse cells under them
    auto const under_of = [&](core::Box<int, dim> const& fine_box) {
        return core::coarsen(core::grow(fine_box, reach), h.ratio);
    };

    // fine patch f's links to the coarse patches under it (only to rank's if mine_only), in
    //  level order: every rank makes the same links for a pair of patches
    auto const under_links = [&](Patch<dim> const& f, bool const mine_only) {
        std::vector<Link<dim>> links;
        auto const footprint = under_of(f.box);
        for (auto const& c : cl.patches)
            for (auto const& shift : coarse_shifts)
                if ((!mine_only or c.rank == rank) and footprint * core::shift(c.box, shift))
                    links.push_back({c, shift, distance(topology, f.rank, c.rank),
                                     local_of(coarse_index, c.id)});
        return links;
    };

    map.under.resize(map.local_fine.size());
    for (std::size_t i = 0; i < map.local_fine.size(); ++i)
        map.under[i] = under_links(map.local_fine[i], false);

    if (!map.local_coarse.empty())
        for (auto const& f : fl.patches)
            if (f.rank != rank)
                if (auto links = under_links(f, true); !links.empty())
                    map.remote.emplace_back(f, std::move(links));
    map.fine_shifts = fine_shifts;

    map.over.resize(map.local_coarse.size());
    for (std::size_t i = 0; i < map.local_coarse.size(); ++i)
    {
        auto const& c = map.local_coarse[i];
        for (auto const& f : fl.patches)
            for (auto const& shift : fine_shifts)
                if (c.box * under_of(core::shift(f.box, shift)))
                    map.over[i].push_back(
                        {f, shift, distance(topology, c.rank, f.rank), local_of(fine_index, f.id)});
    }
    return map;
}

// true if both levels still have the patches the map was built on
template<std::size_t dim>
NO_DISCARD bool is_current(LevelMap<dim> const& map, Hierarchy<dim> const& h)
{
    return level_exists(h, map.fine) and h.levels[map.coarse].patches == map.coarse_built_on
           and h.levels[map.fine].patches == map.fine_built_on;
}

} // namespace damn::amr

#endif // DAMN_AMR_LEVEL_MAP_HPP
