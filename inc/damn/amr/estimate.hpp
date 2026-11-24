#ifndef DAMN_AMR_ESTIMATE_HPP
#define DAMN_AMR_ESTIMATE_HPP

// a decomposition's step time, estimated without data, to choose between decompositions
// (decompose.hpp's partitions, spill tolerances, patch sizes) by what a run would take:
//
//   estimate(h, topology, reach, weights, costs[, threads])
//       per level and rank: compute, its patches' weights (seconds: cost.hpp's CostModel), and
//       exchange, per Distance a latency per message (a patch pair overlapping: another
//       patch's cells, or its periodic image's, within reach of a patch; the coarser level's
//       patches under a patch grown by reach, coarsened) and a cost per cell overlapped; a level
//       takes its slowest rank's compute + exchange, and level l runs ratio^l times a coarse
//       step:  step = sum_l ratio^l max_r (compute[r] + exchange[r])
//   ExchangeCosts{latency, per_cell}, per Distance, seconds
//   exchange_costs(node_latency, node_s_per_byte, bytes_per_cell)
//       the Node distance's from a measurement, the others relative to it (Self, Local 0;
//       Numa 0.3; Socket 2; Remote 10): placeholders until measured
//   Threads{per_rank, main_busy, efficiency}
//       a rank's threads: its patches' tiles (TileSet, a patch's cost shared by its tiles' cells),
//       largest first, each to the least loaded thread (the main one has 1 - main_busy of its
//       time for tiles), a rank's compute its busiest thread's over efficiency; one thread: its
//       patches' sum over 1 - main_busy. Fewer ranks with more threads: fewer exchanges, the
//       imbalance of whole patches within a rank gone, at the threads' efficiency
//   bootstrapped_exchange_costs(bytes_per_cell)
//       from this machine's bootstrap (etc/bootstrap/mpi/exchange_*: Node, Socket where
//       measured), else defaults; bytes_per_cell: what a model exchanges per ghost cell
//
//   Detached{near_band}: Estimate::detached, the step with level 0's far patches (beyond
//       near_band cells of level 1, needed only by the end of the coarse step) as background
//       work on a rank's spare threads: max over ranks of the larger of its far part and the
//       rest. Only with threads (per_rank > 1): with one thread a rank, far level 0 stays in
//       level 0's phase (collectives, dumps need every rank there), Estimate::step
//   choose(h, from, boxes, candidates, reach, costs[, threads])
//       each candidate (a Decompose: partition, spill, sizes, ...) decomposes the levels, each
//       decomposition estimated, the cheapest kept (ties: the first)
//
// An estimate, not a prediction: no overlap of compute and exchange, every exchange of a step
// folded into bytes_per_cell; good enough to rank decompositions

#include "damn/core/def.hpp"
#include "damn/core/def/bootstrap.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/amr/decompose.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/metrics.hpp"
#include "damn/amr/topology.hpp"
#include "damn/core/data/tiles/tile_set.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <queue>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>
#include <iterator>

namespace damn::amr
{
struct ExchangeCosts
{
    std::array<double, n_distances> latency{};  // seconds a message, per Distance
    std::array<double, n_distances> per_cell{}; // seconds a cell, per Distance
};

// the Node distance's costs measured, the others relative to them
NO_DISCARD inline ExchangeCosts
exchange_costs(double const node_latency, double const node_s_per_byte, double const bytes_per_cell)
{
    // Self, Local, Numa, Node, Socket, Remote
    std::array<double, n_distances> constexpr relative{0., 0., 0.3, 1., 2., 10.};
    ExchangeCosts c;
    for (std::size_t d = 0; d < n_distances; ++d)
    {
        c.latency[d]  = relative[d] * node_latency;
        c.per_cell[d] = relative[d] * node_s_per_byte * bytes_per_cell;
    }
    return c;
}

// this machine's, from its bootstrap (Node, and Socket where measured), else defaults (a
// shared memory exchange of ~1 us latency, 1e-10 s a byte)
NO_DISCARD inline ExchangeCosts bootstrapped_exchange_costs(double const bytes_per_cell)
{
    auto c = exchange_costs(bootstrap::get<double>("mpi/exchange_node/latency_s", 1e-6),
                            bootstrap::get<double>("mpi/exchange_node/s_per_byte", 1e-10),
                            bytes_per_cell);
    auto const socket = static_cast<std::size_t>(Distance::Socket);
    c.latency[socket]  = bootstrap::get<double>("mpi/exchange_socket/latency_s", c.latency[socket]);
    c.per_cell[socket] = bootstrap::get<double>("mpi/exchange_socket/s_per_byte",
                                                c.per_cell[socket] / bytes_per_cell)
                         * bytes_per_cell;
    return c;
}

struct Threads
{
    int per_rank      = 1;
    double main_busy  = 0.; // the main thread's share not on tiles (MPI)
    double efficiency = 1.; // the threads' parallel efficiency (shared bandwidth), > 1 thread
};

// level 0's detached (far) part: its patches beyond near_band cells of level 1, needed only by
//  the end of the coarse step, run beside the rest on a rank's spare threads; near_band < 0:
//  not modelled
struct Detached
{
    int near_band = -1;
};

struct LevelEstimate
{
    int level = 0;
    std::vector<double> compute{};  // per rank, seconds
    std::vector<double> exchange{}; // per rank, seconds
    double time = 0;                // the slowest rank's compute + exchange
};

struct Estimate
{
    std::vector<LevelEstimate> levels{};
    double step     = 0; // sum over levels of ratio^level times its time
    double detached = 0; // Detached and threads: far level 0 in the background, else 0
};

namespace detail
{
    // per rank, its patches' compute over its threads (Threads)
    template<std::size_t dim>
    std::vector<double> compute_times(std::vector<Patch<dim>> const& patches,
                                      std::vector<double> const& w, std::size_t const n_ranks,
                                      Threads const& threads)
    {
        std::vector<double> out(n_ranks, 0.);
        if (threads.per_rank <= 1)
        {
            for (std::size_t i = 0; i < patches.size(); ++i)
                out[static_cast<std::size_t>(patches[i].rank)] += w[i];
            for (auto& x : out)
                x /= 1. - threads.main_busy;
            return out;
        }
        // tiles: (rank, cost), patch after patch, a patch's cost shared by its tiles' cells
        std::vector<std::pair<int, double>> tiles;
        for (std::size_t i = 0; i < patches.size(); ++i)
        {
            core::TileSet<core::Box<int, dim>> const set{patches[i].box};
            auto const cells = static_cast<double>(patches[i].box.size());
            for (auto const& tile : set())
                tiles.emplace_back(patches[i].rank,
                                   w[i] * static_cast<double>(tile.size()) / cells);
        }
        // per rank, largest first (ties: tile order)
        std::stable_sort(tiles.begin(), tiles.end(), [](auto const& a, auto const& b) {
            return a.first != b.first ? a.first < b.first : a.second > b.second;
        });
        auto const n_threads = static_cast<std::size_t>(threads.per_rank);
        std::vector<double> speed(n_threads, 1.);
        speed[0]      = 1. - threads.main_busy;
        using Load    = std::pair<double, std::size_t>; // load, thread: least first, then thread
        std::size_t k = 0;
        for (std::size_t r = 0; r < n_ranks; ++r)
        {
            std::priority_queue<Load, std::vector<Load>, std::greater<Load>> heap;
            for (std::size_t t = speed[0] <= 0. ? 1 : 0; t < n_threads; ++t)
                heap.push({0., t});
            for (; k < tiles.size() and static_cast<std::size_t>(tiles[k].first) == r; ++k)
            {
                auto const [load, t] = heap.top();
                heap.pop();
                heap.push({load + tiles[k].second / speed[t], t});
            }
            double busiest = 0.;
            for (; !heap.empty(); heap.pop())
                busiest = std::max(busiest, heap.top().first);
            out[r] = busiest / threads.efficiency;
        }
        return out;
    }

    // per rank, the exchange of the patches' overlaps with `others` (their images by `shifts`)
    //  within each patch's `around` box
    template<std::size_t dim>
    void add_exchange(std::vector<double>& exchange, std::vector<Patch<dim>> const& patches,
                      std::vector<core::Box<int, dim>> const& around,
                      std::vector<Patch<dim>> const& others,
                      std::vector<core::Point<int, dim>> const& shifts, bool const same_level,
                      Topology const& t, ExchangeCosts const& c)
    {
        std::vector<core::Box<int, dim>> boxes;
        boxes.reserve(others.size());
        for (auto const& o : others)
            boxes.push_back(o.box);
        PatchIndex<dim> index{boxes};
        for (std::size_t i = 0; i < patches.size(); ++i)
        {
            auto const& p = patches[i];
            for (auto const& s : shifts)
            {
                // the others' images shifted by s inside around: the others inside around - s
                auto const query = core::shift(around[i], s * -1);
                index.overlaps(query, [&](std::size_t const j, double const cells) {
                    auto const& o   = others[j];
                    bool const self = same_level and o.id == p.id;
                    if (self and s == core::Point<int, dim>{})
                        return; // its own cells
                    auto const d = static_cast<std::size_t>(self ? Distance::Self
                                                                 : distance(t, p.rank, o.rank));
                    exchange[static_cast<std::size_t>(p.rank)]
                        += c.latency[d] + c.per_cell[d] * cells;
                });
            }
        }
    }
} // namespace detail

// weights[ln]: one per patch of level ln, seconds (empty or missing: its cells, seconds a cell
// of 1); reach in each level's cells
template<std::size_t dim>
NO_DISCARD Estimate estimate(Hierarchy<dim> const& h, Topology const& t, int const reach,
                             std::vector<std::vector<double>> const& weights,
                             ExchangeCosts const& costs, Threads const& threads = {},
                             Detached const& detached = {})
{
    if (reach < 0)
        throw std::runtime_error("damn::amr::estimate: negative reach");
    auto const n_ranks = t.ranks.size();
    Estimate out;
    double substeps = 1;
    for (std::size_t ln = 0; ln < h.levels.size(); ++ln)
    {
        auto const& level = h.levels[ln];
        auto const lvl    = static_cast<int>(ln);
        bool const given  = ln < weights.size() and !weights[ln].empty();
        if (given and weights[ln].size() != level.patches.size())
            throw std::runtime_error("damn::amr::estimate: one weight per patch of level "
                                     + std::to_string(ln));
        LevelEstimate e;
        e.level = lvl;
        e.exchange.assign(n_ranks, 0.);
        std::vector<core::Box<int, dim>> grown;
        std::vector<double> w;
        for (std::size_t i = 0; i < level.patches.size(); ++i)
        {
            auto const& p = level.patches[i];
            if (p.rank < 0 or static_cast<std::size_t>(p.rank) >= n_ranks)
                throw std::runtime_error("damn::amr::estimate: a patch of rank "
                                         + std::to_string(p.rank) + " outside the topology");
            w.push_back(given ? weights[ln][i] : static_cast<double>(p.box.size()));
            grown.push_back(core::grow(p.box, reach));
        }
        e.compute   = detail::compute_times(level.patches, w, n_ranks, threads);
        auto shifts = periodic_shifts(h, lvl);
        shifts.insert(shifts.begin(), core::Point<int, dim>{});
        detail::add_exchange(e.exchange, level.patches, grown, level.patches, shifts, true, t,
                             costs);
        if (ln > 0)
        {
            std::vector<core::Box<int, dim>> under;
            for (auto const& g : grown)
                under.push_back(core::coarsen(g, h.ratio));
            auto coarse_shifts = periodic_shifts(h, lvl - 1);
            coarse_shifts.insert(coarse_shifts.begin(), core::Point<int, dim>{});
            detail::add_exchange(e.exchange, level.patches, under, h.levels[ln - 1].patches,
                                 coarse_shifts, false, t, costs);
        }
        for (std::size_t r = 0; r < n_ranks; ++r)
            e.time = std::max(e.time, e.compute[r] + e.exchange[r]);
        out.step += substeps * e.time;
        substeps *= h.ratio;
        out.levels.push_back(std::move(e));
    }
    if (detached.near_band >= 0 and threads.per_rank > 1 and !out.levels.empty())
    {
        // per rank its whole step, and level 0's far part of it
        std::vector<double> total(n_ranks, 0.), far(n_ranks, 0.);
        double sub = 1;
        for (auto const& e : out.levels)
        {
            for (std::size_t r = 0; r < n_ranks; ++r)
                total[r] += sub * (e.compute[r] + e.exchange[r]);
            sub *= h.ratio;
        }
        if (h.levels.size() > 1)
        {
            std::vector<core::Box<int, dim>> near;
            for (auto const& p : h.levels[1].patches)
                near.push_back(core::grow(core::coarsen(p.box, h.ratio), detached.near_band));
            auto const& level0 = h.levels[0].patches;
            for (std::size_t i = 0; i < level0.size(); ++i)
            {
                bool const is_near = std::any_of(near.begin(), near.end(), [&](auto const& b) {
                    return static_cast<bool>(b * level0[i].box);
                });
                if (!is_near)
                    far[static_cast<std::size_t>(level0[i].rank)]
                        += (!weights.empty() and !weights[0].empty()
                                ? weights[0][i]
                                : static_cast<double>(level0[i].box.size()))
                           / (1. - threads.main_busy);
            }
        }
        for (std::size_t r = 0; r < n_ranks; ++r)
            out.detached = std::max(out.detached, std::max(total[r] - far[r], far[r]));
    }
    return out;
}

template<std::size_t dim>
struct Choice
{
    std::size_t candidate = 0;        // the cheapest candidate's index (ties: the first)
    std::vector<Level<dim>> levels{}; // its levels from `from`
    std::vector<double> steps{};      // every candidate's estimated step
};

// levels from, from + 1, ... decomposed by each candidate (decompose), each estimated (its own
// weigh on its patches, h's levels under `from` kept), the cheapest kept
template<std::size_t dim>
NO_DISCARD Choice<dim> choose(Hierarchy<dim> const& h, int const from,
                              std::vector<std::vector<core::Box<int, dim>>> const& boxes,
                              std::vector<Decompose<dim>> const& candidates, int const reach,
                              ExchangeCosts const& costs, Threads const& threads = {})
{
    if (candidates.empty())
        throw std::runtime_error("damn::amr::choose: no candidates");
    Choice<dim> out;
    double best = 0;
    for (std::size_t c = 0; c < candidates.size(); ++c)
    {
        auto levels = decompose(h, from, boxes, candidates[c]);
        auto with   = h;
        with.levels.resize(static_cast<std::size_t>(from));
        for (auto const& l : levels)
            with.levels.push_back(l);
        std::vector<std::vector<double>> weights;
        for (auto const& l : with.levels)
        {
            std::vector<core::Box<int, dim>> bs;
            for (auto const& p : l.patches)
                bs.push_back(p.box);
            weights.push_back(candidates[c].weigh(l.number, bs));
        }
        auto const step
            = estimate(with, candidates[c].topology, reach, weights, costs, threads).step;
        out.steps.push_back(step);
        if (c == 0 or step < best)
        {
            best          = step;
            out.candidate = c;
            out.levels    = std::move(levels);
        }
    }
    return out;
}

} // namespace damn::amr

#endif // DAMN_AMR_ESTIMATE_HPP
