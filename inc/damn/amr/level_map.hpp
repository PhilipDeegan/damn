#ifndef DAMN_AMR_LEVEL_MAP_HPP
#define DAMN_AMR_LEVEL_MAP_HPP

// LevelMap: how the patches of two consecutive levels overlap, periodic images included
//
//   under[i]: for the i-th local fine patch, the coarse patches under its box grown by reach
//             (fine cells), e.g. to fill its level border ghosts or a new patch from coarse
//             data; link shifts are in the coarse index space
//   over[i]:  for the i-th local coarse patch, the fine patches over it, e.g. to coarsen fine
//             data onto it; link shifts are in the fine index space
//
// purely geometric, as Neighbours: built per regrid of either level. Refining and coarsening
// data along it come later

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
#include <stdexcept>
#include <string>
#include <vector>

namespace damn::amr
{
template<std::size_t dim>
struct LevelMap
{
    int coarse = 0; // level numbers, fine == coarse + 1
    int fine   = 1;
    int rank   = 0;
    int reach  = 0; // fine cells
    std::vector<Patch<dim>> coarse_built_on{}, fine_built_on{};
    std::vector<Patch<dim>> local_coarse{}, local_fine{}; // this rank's, in LevelData order
    std::vector<std::vector<Link<dim>>> under{};          // per local fine patch
    std::vector<std::vector<Link<dim>>> over{};           // per local coarse patch
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

    LevelMap<dim> map{coarse,
                      fine,
                      rank,
                      reach,
                      cl.patches,
                      fl.patches,
                      local_patches(cl, rank),
                      local_patches(fl, rank),
                      {},
                      {}};

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

    map.under.resize(map.local_fine.size());
    for (std::size_t i = 0; i < map.local_fine.size(); ++i)
    {
        auto const& f        = map.local_fine[i];
        auto const footprint = under_of(f.box);
        for (auto const& c : cl.patches)
            for (auto const& shift : coarse_shifts)
                if (footprint * core::shift(c.box, shift))
                    map.under[i].push_back({c, shift, distance(topology, f.rank, c.rank),
                                            local_of(coarse_index, c.id)});
    }

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
