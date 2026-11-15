#ifndef DAMN_AMR_PARTITION_HPP
#define DAMN_AMR_PARTITION_HPP

// load balance: boxes (patches) to ranks along a Hilbert space filling curve
//
//   partition(boxes, weights, n_ranks)  rank of each box: the curve cut into n_ranks runs of
//                                       about equal weight
//
// the curve keeps a rank's boxes together, so most neighbours are on the same rank. A box goes to
// the rank whose share of the total weight holds the middle of the box's weight along the curve:
// every rank ends within one box's weight of total / n_ranks, and the ranks are contiguous runs
// of the curve. Deterministic: every rank computes the same partition from the same input
// weights: whatever costs, e.g. cells, or particles (density above a cutoff times particles per
// cell)
//
//   cut_partition(boxes, block, block_weights, n_ranks)  boxes and their ranks, boxes cut where a
//                                       rank's share ends inside them
//
// as SAMRAI's CascadePartitioner, but along the curve: patches can be large (tiled), so one box's
// weight may be more than a rank's tolerance. A box is cut into blocks (blocks_of: `block` cells
// from its lower corner, the remainder in the last block of each direction); a box a rank's share
// ends in is split by its blocks along the curve, and the blocks of each rank coalesced back into
// boxes. Weights are per block (blocks(boxes, block): box after box), the same on every rank,
// e.g. particles counted per block on the rank that holds the box then summed over the ranks

#include "damn/core/def.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/logger.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace damn::amr
{
// index along the Hilbert curve of a point in [0, 2^bits)^dim (dim * bits <= 64), from J. Skilling,
// "Programming the Hilbert curve", AIP Conf. Proc. 707, 381 (2004): consecutive indices are
// face neighbours
template<std::size_t dim>
NO_DISCARD std::uint64_t hilbert_index(std::array<std::uint32_t, dim> x, unsigned const bits);

// the order of boxes along the curve, by their centres in box's index space
template<std::size_t dim>
NO_DISCARD std::vector<std::size_t> curve_order(std::vector<core::Box<int, dim>> const& boxes);

// each box's rank in [0, n_ranks)
template<std::size_t dim>
NO_DISCARD std::vector<int> partition(std::vector<core::Box<int, dim>> const& boxes,
                                      std::vector<double> const& weights, int n_ranks);

// a box's cells, the weight of field work
template<std::size_t dim>
NO_DISCARD std::vector<double> cell_weights(std::vector<core::Box<int, dim>> const& boxes);

// a and b as one box, if their union is one
template<std::size_t dim>
NO_DISCARD std::optional<core::Box<int, dim>> union_box(core::Box<int, dim> const& a,
                                                        core::Box<int, dim> const& b);

// boxes merged two by two while their union is a box, deterministic (lower corner order)
template<std::size_t dim>
NO_DISCARD std::vector<core::Box<int, dim>> coalesce(std::vector<core::Box<int, dim>> boxes);

// box cut from its lower corner into blocks of `block` cells per direction, the remainder in the
// last block (block to 2 * block - 1 cells, or the box's width if less), x fastest
template<std::size_t dim>
NO_DISCARD std::vector<core::Box<int, dim>> blocks_of(core::Box<int, dim> const& box,
                                                      std::array<int, dim> const& block);

// every box's blocks, box after box: the order of cut_partition's block weights
template<std::size_t dim>
NO_DISCARD std::vector<core::Box<int, dim>> blocks(std::vector<core::Box<int, dim>> const& boxes,
                                                   std::array<int, dim> const& block);

template<std::size_t dim>
struct Partition
{
    std::vector<core::Box<int, dim>> boxes; // along the curve
    std::vector<int> ranks;                 // non decreasing
};

// boxes partitioned over n_ranks, a box cut on its blocks where a rank's share ends inside it,
// unless less than tol of a share is on one side (the box then goes whole to the rank holding the
// middle of its weight). Each rank ends within a block's weight of total / n_ranks, plus tol of a
// share per end of its run left uncut
template<std::size_t dim>
NO_DISCARD Partition<dim>
cut_partition(std::vector<core::Box<int, dim>> const& boxes, std::array<int, dim> const& block,
              std::vector<double> const& block_weights, int n_ranks, double tol = 0.05);


template<std::size_t dim>
std::uint64_t hilbert_index(std::array<std::uint32_t, dim> x, unsigned const bits)
{
    static_assert(dim > 0 and dim < 4);
    if (bits == 0)
        return 0;
    if (dim * bits > 64)
        throw std::runtime_error("damn::amr::hilbert_index: too many bits");

    std::uint32_t const top = 1u << (bits - 1);

    // inverse undo
    for (std::uint32_t q = top; q > 1; q >>= 1)
    {
        std::uint32_t const p = q - 1;
        for (std::size_t i = 0; i < dim; ++i)
            if (x[i] & q)
                x[0] ^= p;
            else
            {
                std::uint32_t const t = (x[0] ^ x[i]) & p;
                x[0] ^= t;
                x[i] ^= t;
            }
    }
    // Gray encode
    for (std::size_t i = 1; i < dim; ++i)
        x[i] ^= x[i - 1];
    std::uint32_t t = 0;
    for (std::uint32_t q = top; q > 1; q >>= 1)
        if (x[dim - 1] & q)
            t ^= q - 1;
    for (std::size_t i = 0; i < dim; ++i)
        x[i] ^= t;

    // the transposed index: bit b of x[i] is bit b * dim + (dim - 1 - i) of the index
    std::uint64_t h = 0;
    for (int b = static_cast<int>(bits) - 1; b >= 0; --b)
        for (std::size_t i = 0; i < dim; ++i)
            h = (h << 1) | ((x[i] >> b) & 1u);
    return h;
}

template<std::size_t dim>
std::vector<std::size_t> curve_order(std::vector<core::Box<int, dim>> const& boxes)
{
    std::vector<std::size_t> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    if (boxes.empty())
        return order;

    // twice the centres (integers), from the lowest corner of all boxes
    std::array<std::int64_t, dim> lo{}, hi{};
    lo.fill(std::numeric_limits<std::int64_t>::max());
    hi.fill(std::numeric_limits<std::int64_t>::min());
    for (auto const& b : boxes)
        for (std::size_t i = 0; i < dim; ++i)
        {
            lo[i] = std::min<std::int64_t>(lo[i], 2 * b.lower[i]);
            hi[i] = std::max<std::int64_t>(hi[i], 2 * b.upper[i] + 2);
        }

    unsigned bits = 1;
    for (std::size_t i = 0; i < dim; ++i)
        while ((std::int64_t{1} << bits) <= hi[i] - lo[i])
            ++bits;
    if (dim * bits > 64)
        throw std::runtime_error("damn::amr::curve_order: index space too large");

    std::vector<std::uint64_t> key(boxes.size());
    for (std::size_t k = 0; k < boxes.size(); ++k)
    {
        std::array<std::uint32_t, dim> c{};
        for (std::size_t i = 0; i < dim; ++i)
            c[i] = static_cast<std::uint32_t>(boxes[k].lower[i] + boxes[k].upper[i] + 1 - lo[i]);
        key[k] = hilbert_index(c, bits);
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](auto const a, auto const b) { return key[a] < key[b]; });
    return order;
}

template<std::size_t dim>
std::vector<int> partition(std::vector<core::Box<int, dim>> const& boxes,
                           std::vector<double> const& weights, int const n_ranks)
{
    DAMN_LOG_SCOPE(1, "damn::amr::partition");
    if (n_ranks < 1)
        throw std::runtime_error("damn::amr::partition: n_ranks < 1");
    if (weights.size() != boxes.size())
        throw std::runtime_error("damn::amr::partition: one weight per box");
    for (auto const w : weights)
        if (!(w >= 0))
            throw std::runtime_error("damn::amr::partition: weights must be >= 0");

    std::vector<int> ranks(boxes.size(), 0);
    auto const total = std::accumulate(weights.begin(), weights.end(), 0.);
    if (n_ranks == 1 or boxes.empty())
        return ranks;

    auto const order = curve_order(boxes);
    if (total <= 0) // nothing weighs: as many boxes per rank
    {
        for (std::size_t k = 0; k < order.size(); ++k)
            ranks[order[k]] = static_cast<int>(k * n_ranks / order.size());
        return ranks;
    }

    double const share = total / n_ranks;
    double before      = 0;
    for (auto const k : order)
    {
        auto const middle = before + weights[k] / 2;
        ranks[k]          = std::min(n_ranks - 1, static_cast<int>(middle / share));
        before += weights[k];
    }
    return ranks;
}

template<std::size_t dim>
std::vector<double> cell_weights(std::vector<core::Box<int, dim>> const& boxes)
{
    std::vector<double> weights;
    weights.reserve(boxes.size());
    for (auto const& b : boxes)
        weights.push_back(static_cast<double>(b.size()));
    return weights;
}


template<std::size_t dim>
std::optional<core::Box<int, dim>> union_box(core::Box<int, dim> const& a,
                                             core::Box<int, dim> const& b)
{
    std::size_t differ = dim;
    for (std::size_t i = 0; i < dim; ++i)
        if (a.lower[i] != b.lower[i] or a.upper[i] != b.upper[i])
        {
            if (differ != dim)
                return std::nullopt; // more than one direction
            differ = i;
        }
    if (differ == dim)
        return a;
    auto const i = differ;
    if (a.upper[i] + 1 != b.lower[i] and b.upper[i] + 1 != a.lower[i])
        return std::nullopt; // not adjacent
    return a.merge(b);
}

template<std::size_t dim>
std::vector<core::Box<int, dim>> coalesce(std::vector<core::Box<int, dim>> boxes)
{
    bool merged = true;
    while (merged)
    {
        merged = false;
        std::sort(boxes.begin(), boxes.end(), [](auto const& a, auto const& b) {
            auto const al = a.lower.toArray(), bl = b.lower.toArray();
            return al < bl or (al == bl and a.upper.toArray() < b.upper.toArray());
        });
        for (std::size_t i = 0; i < boxes.size() and !merged; ++i)
            for (std::size_t j = i + 1; j < boxes.size(); ++j)
                if (auto const u = union_box(boxes[i], boxes[j]))
                {
                    boxes[i] = *u;
                    boxes.erase(boxes.begin() + static_cast<std::ptrdiff_t>(j));
                    merged = true;
                    break;
                }
    }
    return boxes;
}

template<std::size_t dim>
std::vector<core::Box<int, dim>> blocks_of(core::Box<int, dim> const& box,
                                           std::array<int, dim> const& block)
{
    std::array<int, dim> n{};
    std::size_t total = 1;
    for (std::size_t i = 0; i < dim; ++i)
    {
        if (block[i] < 1)
            throw std::runtime_error("damn::amr::blocks_of: block < 1");
        n[i] = std::max(1, (box.upper[i] - box.lower[i] + 1) / block[i]);
        total *= static_cast<std::size_t>(n[i]);
    }

    std::vector<core::Box<int, dim>> out;
    out.reserve(total);
    for (std::size_t flat = 0; flat < total; ++flat)
    {
        auto rest = flat;
        core::Box<int, dim> b{box};
        for (std::size_t i = 0; i < dim; ++i)
        {
            auto const k = static_cast<int>(rest % static_cast<std::size_t>(n[i]));
            rest /= static_cast<std::size_t>(n[i]);
            b.lower[i] = box.lower[i] + k * block[i];
            b.upper[i] = k + 1 == n[i] ? box.upper[i] : b.lower[i] + block[i] - 1;
        }
        out.push_back(b);
    }
    return out;
}

template<std::size_t dim>
std::vector<core::Box<int, dim>> blocks(std::vector<core::Box<int, dim>> const& boxes,
                                        std::array<int, dim> const& block)
{
    std::vector<core::Box<int, dim>> out;
    for (auto const& box : boxes)
        for (auto const& b : blocks_of(box, block))
            out.push_back(b);
    return out;
}

template<std::size_t dim>
Partition<dim>
cut_partition(std::vector<core::Box<int, dim>> const& boxes, std::array<int, dim> const& block,
              std::vector<double> const& block_weights, int const n_ranks, double const tol)
{
    DAMN_LOG_SCOPE(1, "damn::amr::cut_partition");
    if (n_ranks < 1)
        throw std::runtime_error("damn::amr::cut_partition: n_ranks < 1");
    if (!(tol >= 0))
        throw std::runtime_error("damn::amr::cut_partition: tol < 0");
    for (auto const w : block_weights)
        if (!(w >= 0))
            throw std::runtime_error("damn::amr::cut_partition: weights must be >= 0");

    // each box's blocks, their first weight and the box's weight
    std::vector<std::vector<core::Box<int, dim>>> box_blocks;
    std::vector<std::size_t> first;
    std::vector<double> weights;
    std::size_t n_blocks = 0;
    for (auto const& box : boxes)
    {
        box_blocks.push_back(blocks_of(box, block));
        first.push_back(n_blocks);
        n_blocks += box_blocks.back().size();
    }
    if (block_weights.size() != n_blocks)
        throw std::runtime_error("damn::amr::cut_partition: one weight per block");
    for (std::size_t k = 0; k < boxes.size(); ++k)
    {
        auto const it = block_weights.begin() + static_cast<std::ptrdiff_t>(first[k]);
        weights.push_back(
            std::accumulate(it, it + static_cast<std::ptrdiff_t>(box_blocks[k].size()), 0.));
    }

    Partition<dim> out;
    auto const order = curve_order(boxes);
    auto const total = std::accumulate(weights.begin(), weights.end(), 0.);
    if (n_ranks == 1 or total <= 0) // nothing to cut: whole boxes
    {
        auto const ranks = partition(boxes, weights, n_ranks);
        for (auto const k : order)
        {
            out.boxes.push_back(boxes[k]);
            out.ranks.push_back(ranks[k]);
        }
        return out;
    }

    double const share = total / n_ranks;
    auto const rank_of = [&](double const middle) {
        return std::min(n_ranks - 1, static_cast<int>(middle / share));
    };

    // a share ends inside [lo, hi) with more than tol of a share on both sides
    auto const worth_cutting = [&](double const lo, double const hi) {
        for (int j = static_cast<int>(lo / share) + 1; j < n_ranks and j * share < hi; ++j)
            if (std::min(j * share - lo, hi - j * share) > tol * share)
                return true;
        return false;
    };

    double before = 0;
    for (auto const k : order)
    {
        if (!worth_cutting(before, before + weights[k]))
        {
            out.boxes.push_back(boxes[k]);
            out.ranks.push_back(rank_of(before + weights[k] / 2));
            before += weights[k];
            continue;
        }

        // the box's blocks along the curve, each to the rank holding its middle
        auto const& bb = box_blocks[k];
        std::vector<std::vector<core::Box<int, dim>>> per_rank(static_cast<std::size_t>(n_ranks));
        for (auto const b : curve_order(bb))
        {
            auto const w = block_weights[first[k] + b];
            per_rank[static_cast<std::size_t>(rank_of(before + w / 2))].push_back(bb[b]);
            before += w;
        }
        for (std::size_t r = 0; r < per_rank.size(); ++r)
            for (auto const& piece : coalesce(std::move(per_rank[r])))
            {
                out.boxes.push_back(piece);
                out.ranks.push_back(static_cast<int>(r));
            }
    }
    return out;
}

} // namespace damn::amr

#endif // DAMN_AMR_PARTITION_HPP
