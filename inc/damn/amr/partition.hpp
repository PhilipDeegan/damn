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

#include "damn/core/def.hpp"
#include "damn/core/logger.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace damn::core
{
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core

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

} // namespace damn::amr

#endif // DAMN_AMR_PARTITION_HPP
