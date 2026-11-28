#ifndef DAMN_AMR_KEEP_PARTITION_HPP
#define DAMN_AMR_KEEP_PARTITION_HPP

// partitions keeping data where it is (the old level's owners, the coarser level's under it),
// spill to rebalance: doc/amr/keep_partition.md

#include "damn/core/def.hpp"
#include "damn/amr/partition.hpp"
#include "damn/amr/topology.hpp"
#include "damn/core/geometry/box.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <initializer_list>
#include <iterator>
#include <string>

namespace damn::amr
{
template<std::size_t dim>
struct Patch;

template<std::size_t dim>
struct OwnedBox
{
    core::Box<int, dim> box{};
    int rank = 0;

    bool operator==(OwnedBox const&) const = default;
};

namespace detail
{
    // the parts of `boxes` outside `cut`
    template<std::size_t dim>
    std::vector<core::Box<int, dim>> minus(std::vector<core::Box<int, dim>> const& boxes,
                                           core::Box<int, dim> const& cut)
    {
        std::vector<core::Box<int, dim>> out;
        for (auto const& b : boxes)
        {
            if (!(b * cut))
            {
                out.push_back(b);
                continue;
            }
            for (auto const& piece : b.remove(cut))
                out.push_back(piece);
        }
        return out;
    }
} // namespace detail

template<std::size_t dim>
MKN_KUL_FN_SIG_RET std::vector<OwnedBox<dim>>
keep(std::vector<core::Box<int, dim>> const& boxes, std::vector<Patch<dim>> const& old,
     std::vector<Patch<dim>> const& coarser, int const ratio, auto&& fresh)
{
    std::vector<OwnedBox<dim>> pieces;
    std::vector<core::Box<int, dim>> rest_all;
    for (auto const& box : boxes)
    {
        std::vector<core::Box<int, dim>> rest{box};
        for (auto const& o : old)
            if (auto const x = box * o.box)
            {
                pieces.push_back({*x, o.rank});
                rest = detail::minus(rest, o.box);
            }
        for (auto const& c : coarser)
        {
            auto const fine = core::refine(c.box, ratio);
            std::vector<core::Box<int, dim>> left;
            for (auto const& part : rest)
            {
                if (auto const x = part * fine)
                {
                    pieces.push_back({*x, c.rank});
                    for (auto const& p : detail::minus<dim>({part}, fine))
                        left.push_back(p);
                }
                else
                    left.push_back(part);
            }
            rest = std::move(left);
        }
        rest_all.insert(rest_all.end(), rest.begin(), rest.end());
    }
    if (!rest_all.empty())
    {
        std::size_t cells = 0, given = 0;
        for (auto const& b : rest_all)
            cells += b.size();
        for (auto const& p : fresh(rest_all))
        {
            given += p.box.size();
            pieces.push_back(p);
        }
        if (given != cells)
            throw std::runtime_error("damn::amr::keep: fresh gave " + std::to_string(given)
                                     + " cells for " + std::to_string(cells));
    }
    return pieces;
}

template<std::size_t dim>
MKN_KUL_FN_SIG_RET std::vector<std::vector<core::Box<int, dim>>>
owned_boxes(std::vector<OwnedBox<dim>> const& pieces, int const n_ranks)
{
    std::vector<std::vector<core::Box<int, dim>>> out(static_cast<std::size_t>(n_ranks));
    for (auto const& p : pieces)
    {
        if (p.rank < 0 or p.rank >= n_ranks)
            throw std::runtime_error("damn::amr::owned_boxes: rank " + std::to_string(p.rank)
                                     + " of " + std::to_string(n_ranks));
        out[static_cast<std::size_t>(p.rank)].push_back(p.box);
    }
    for (auto& boxes : out)
        boxes = coalesce(std::move(boxes));
    return out;
}

template<std::size_t dim>
MKN_KUL_FN_SIG_RET std::vector<int>
spill(std::vector<core::Box<int, dim>> const& boxes, std::vector<int> ranks,
      std::vector<double> const& weights, Topology const& topology, double const tol)
{
    if (ranks.size() != boxes.size() or weights.size() != boxes.size())
        throw std::runtime_error("damn::amr::spill: one rank and one weight per box");
    auto const n        = topology.ranks.size();
    auto const capacity = node_share_capacity(topology);
    auto const cap_sum  = std::accumulate(capacity.begin(), capacity.end(), 0.);
    auto const total    = std::accumulate(weights.begin(), weights.end(), 0.);
    if (total <= 0. or cap_sum <= 0.)
        return ranks;

    std::vector<double> load(n, 0.), limit(n, 0.);
    std::vector<std::vector<std::size_t>> mine(n);
    for (std::size_t i = 0; i < boxes.size(); ++i)
    {
        auto const r = static_cast<std::size_t>(ranks[i]);
        load[r] += weights[i];
        mine[r].push_back(i);
    }
    for (std::size_t r = 0; r < n; ++r)
        limit[r] = (1. + tol) * total * capacity[r] / cap_sum;

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](auto const a, auto const b) { return load[a] > load[b]; });
    for (auto const r : order)
        while (load[r] > limit[r] and mine[r].size() > 1)
        {
            auto const excess = load[r] - limit[r];
            auto& own         = mine[r];
            // the heaviest patch within the excess, else the lightest (ties: lower index)
            std::size_t pick = own.size();
            for (std::size_t k = 0; k < own.size(); ++k)
                if (weights[own[k]] <= excess
                    and (pick == own.size() or weights[own[k]] > weights[own[pick]]))
                    pick = k;
            if (pick == own.size())
                pick = static_cast<std::size_t>(std::distance(
                    own.begin(), std::min_element(own.begin(), own.end(), [&](auto a, auto b) {
                        return weights[a] < weights[b];
                    })));
            auto const i = own[pick];
            auto const w = weights[i];
            // the least loaded rank with room, the closest first: its memory domain (Node), its
            //  node (Socket), any (Remote)
            auto best = n;
            for (auto const within : {Distance::Node, Distance::Socket, Distance::Remote})
            {
                for (std::size_t t = 0; t < n; ++t)
                    if (t != r and load[t] + w <= limit[t]
                        and distance(topology.ranks[r], topology.ranks[t]) <= within
                        and (best == n or load[t] < load[best]))
                        best = t;
                if (best != n)
                    break;
            }
            if (best == n)
                break; // nowhere to go: patch granularity
            own.erase(own.begin() + static_cast<std::ptrdiff_t>(pick));
            mine[best].push_back(i);
            load[r] -= w;
            load[best] += w;
            ranks[i] = static_cast<int>(best);
        }
    return ranks;
}

} // namespace damn::amr

#endif // DAMN_AMR_KEEP_PARTITION_HPP
