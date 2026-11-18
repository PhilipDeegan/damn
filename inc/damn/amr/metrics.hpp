#ifndef DAMN_AMR_METRICS_HPP
#define DAMN_AMR_METRICS_HPP

// what a hierarchy's decomposition costs, without data: per level, how much each rank holds and
// how many cells its patches exchange, by Distance (topology.hpp), for comparing decompositions
// (chopping, partitioning) before a run:
//
//   ghost_cells[d]:  same level, the cells of other patches (and periodic images) inside each
//                    patch's box grown by reach, whose owner is at distance d
//   coarse_cells[d]: levels > 0, the coarser level's cells under each patch's box grown by
//                    reach (coarsened), whose owner is at distance d: what refining its ghosts
//                    reads
//   weight[r]:       rank r's sum of the level's patch weights (cells if none given)
//
// Self counts a patch's own periodic images; Local the same rank's other patches.

#include "damn/core/def.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/topology.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"

#include <array>
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace damn::amr
{
inline constexpr std::size_t n_distances = 5; // Distance's values

struct LevelMetrics
{
    int level                  = 0;
    std::size_t patches        = 0;
    std::size_t cells          = 0;
    std::size_t smallest_patch = 0; // cells
    std::vector<double> weight{};   // per rank
    std::array<std::size_t, n_distances> ghost_cells{};
    std::array<std::size_t, n_distances> coarse_cells{};

    // the busiest rank's weight over the mean (1: balanced)
    NO_DISCARD double imbalance() const
    {
        if (weight.empty())
            return 1.;
        auto const total = std::accumulate(weight.begin(), weight.end(), 0.);
        if (total == 0.)
            return 1.;
        return *std::max_element(weight.begin(), weight.end())
               / (total / static_cast<double>(weight.size()));
    }
};

namespace detail
{
    // the cells of the boxes' images (shifts, the unshifted one included) inside `around`, by
    // the distance from `rank` to each box's owner (Self: the patch `self` itself)
    template<std::size_t dim>
    void add_overlaps(std::array<std::size_t, n_distances>& out, Patch<dim> const& from,
                      core::Box<int, dim> const& around, std::vector<Patch<dim>> const& patches,
                      std::vector<core::Point<int, dim>> const& shifts, Topology const& t,
                      bool const same_level)
    {
        for (auto const& p : patches)
            for (auto const& s : shifts)
            {
                bool const self = same_level and p.id == from.id;
                if (self and s == core::Point<int, dim>{})
                    continue; // its own cells
                if (auto const o = around * core::shift(p.box, s))
                {
                    auto const d = self ? Distance::Self : distance(t, from.rank, p.rank);
                    out[static_cast<std::size_t>(d)] += o->size();
                }
            }
    }
} // namespace detail


// per level of h: weight per rank, ghost and coarse cells exchanged by distance; weights[ln]: one
// per patch of level ln (empty or missing: the patch's cells); reach in each level's cells
template<std::size_t dim>
NO_DISCARD std::vector<LevelMetrics> metrics(Hierarchy<dim> const& h, Topology const& t,
                                             int const reach,
                                             std::vector<std::vector<double>> const& weights = {})
{
    if (reach < 0)
        throw std::runtime_error("damn::amr::metrics: negative reach");
    auto const n_ranks = t.ranks.size();

    std::vector<LevelMetrics> out;
    for (std::size_t ln = 0; ln < h.levels.size(); ++ln)
    {
        auto const& level = h.levels[ln];
        auto const lvl    = static_cast<int>(ln);
        LevelMetrics m;
        m.level   = lvl;
        m.patches = level.patches.size();
        m.weight.assign(n_ranks, 0.);
        m.smallest_patch = level.patches.empty() ? 0 : level.patches.front().box.size();

        bool const weighted = ln < weights.size() and !weights[ln].empty();
        if (weighted and weights[ln].size() != level.patches.size())
            throw std::runtime_error("damn::amr::metrics: one weight per patch of level "
                                     + std::to_string(ln));

        auto shifts = periodic_shifts(h, lvl);
        shifts.insert(shifts.begin(), core::Point<int, dim>{});
        std::vector<core::Point<int, dim>> coarse_shifts;
        if (ln > 0)
        {
            coarse_shifts = periodic_shifts(h, lvl - 1);
            coarse_shifts.insert(coarse_shifts.begin(), core::Point<int, dim>{});
        }

        for (std::size_t i = 0; i < level.patches.size(); ++i)
        {
            auto const& p = level.patches[i];
            if (p.rank < 0 or static_cast<std::size_t>(p.rank) >= n_ranks)
                throw std::runtime_error("damn::amr::metrics: a patch of rank "
                                         + std::to_string(p.rank) + " outside the topology");
            auto const cells = p.box.size();
            m.cells += cells;
            m.smallest_patch = std::min(m.smallest_patch, cells);
            m.weight[p.rank] += weighted ? weights[ln][i] : static_cast<double>(cells);

            auto const grown = core::grow(p.box, reach);
            detail::add_overlaps(m.ghost_cells, p, grown, level.patches, shifts, t, true);
            if (ln > 0)
                detail::add_overlaps(m.coarse_cells, p, core::coarsen(grown, h.ratio),
                                     h.levels[ln - 1].patches, coarse_shifts, t, false);
        }
        out.push_back(std::move(m));
    }
    return out;
}

// what moving from hierarchy `from` to `to` (a regrid, a rebalance) costs, without data, per level
// of `to`, by Distance from the old owner to the new one:
//
//   moved[d]:   cells of a new patch the old level held (any patch), owned at distance d
//   refined[d]: cells of a new patch the old level did not hold, refined from the old coarser
//               level's cells under them (coarsened), whose owner is at distance d (level 0: none,
//               it always covers the domain)
//
// Self and Local are free (the data stays on its rank); the rest is copied between ranks
struct LevelMigration
{
    int level = 0;
    std::array<std::size_t, n_distances> moved{};
    std::array<std::size_t, n_distances> refined{};
};

template<std::size_t dim>
NO_DISCARD std::vector<LevelMigration> migration(Hierarchy<dim> const& from,
                                                 Hierarchy<dim> const& to, Topology const& t)
{
    if (from.ratio != to.ratio)
        throw std::runtime_error("damn::amr::migration: different refinement ratios");
    auto const at = [](std::array<std::size_t, n_distances>& a, Distance const d) -> auto& {
        return a[static_cast<std::size_t>(d)];
    };
    auto const local = [](Distance const d) { return d == Distance::Self ? Distance::Local : d; };

    std::vector<LevelMigration> out;
    for (std::size_t ln = 0; ln < to.levels.size(); ++ln)
    {
        LevelMigration m;
        m.level                = static_cast<int>(ln);
        bool const had         = ln < from.levels.size();
        bool const has_coarser = ln > 0 and ln - 1 < from.levels.size();
        for (auto const& p : to.levels[ln].patches)
        {
            std::size_t held = 0;
            if (had)
                for (auto const& o : from.levels[ln].patches)
                    if (auto const overlap = p.box * o.box)
                    {
                        held += overlap->size();
                        at(m.moved, local(distance(t, o.rank, p.rank))) += overlap->size();
                    }
            if (held == p.box.size() or !has_coarser)
                continue;
            // the cells the old level did not hold: from the old coarser level under them, a
            //  coarse cell for ratio^dim fine ones (whole coarse cells: levels are)
            std::size_t fine_per_coarse = 1;
            for (std::size_t i = 0; i < dim; ++i)
                fine_per_coarse *= static_cast<std::size_t>(to.ratio);
            auto const coarse = core::coarsen(p.box, to.ratio);
            for (auto const& o : from.levels[ln - 1].patches)
                if (auto const under = coarse * o.box)
                {
                    // the part of the coarse overlap the old level ln did not cover
                    auto fine = under->size() * fine_per_coarse;
                    if (had)
                        for (auto const& prev : from.levels[ln].patches)
                            if (auto const kept = core::refine(*under, to.ratio) * prev.box)
                                if (auto const in_p = *kept * p.box)
                                    fine -= in_p->size();
                    at(m.refined, local(distance(t, o.rank, p.rank))) += fine;
                }
        }
        out.push_back(m);
    }
    return out;
}

} // namespace damn::amr

#endif // DAMN_AMR_METRICS_HPP
