// damn::amr::partition: the Hilbert curve visits every point once through face neighbours,
// partitions are contiguous runs of the curve within one box of the even share, keep neighbours
// together better than lexicographic strips, and follow the weights (PHARE's load balancing
// test: a row of cells with no particles of one population)

#include "damn/amr/partition.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/core/geometry/box_spans.hpp"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <set>
#include <vector>

namespace damn::amr
{
namespace
{
    template<std::size_t dim>
    using Box = core::Box<int, dim>;

    // every point of a 2^bits grid, by its Hilbert index
    template<std::size_t dim>
    std::vector<std::array<std::uint32_t, dim>> by_index(unsigned const bits)
    {
        std::size_t const side = std::size_t{1} << bits;
        std::size_t n          = 1;
        for (std::size_t i = 0; i < dim; ++i)
            n *= side;
        std::vector<std::array<std::uint32_t, dim>> points(n);
        std::vector<bool> seen(n, false);
        for (std::size_t flat = 0; flat < n; ++flat)
        {
            std::array<std::uint32_t, dim> p{};
            auto rest = flat;
            for (std::size_t i = 0; i < dim; ++i)
            {
                p[i] = static_cast<std::uint32_t>(rest % side);
                rest /= side;
            }
            auto const h = hilbert_index(p, bits);
            EXPECT_LT(h, n);
            EXPECT_FALSE(seen[h]) << "index " << h << " twice";
            seen[h]   = true;
            points[h] = p;
        }
        return points;
    }

    std::vector<double> rank_weights(std::vector<int> const& ranks,
                                     std::vector<double> const& weights, int const n_ranks)
    {
        std::vector<double> per_rank(n_ranks, 0);
        for (std::size_t k = 0; k < ranks.size(); ++k)
            per_rank[ranks[k]] += weights[k];
        return per_rank;
    }

    // faces between boxes on different ranks, in cells
    std::size_t cut_faces(std::vector<Box<2>> const& boxes, std::vector<int> const& ranks)
    {
        std::size_t cut = 0;
        for (std::size_t a = 0; a < boxes.size(); ++a)
            for (std::size_t b = a + 1; b < boxes.size(); ++b)
            {
                if (ranks[a] == ranks[b])
                    continue;
                if (auto const o = core::grow(boxes[a], 1) * boxes[b])
                {
                    auto const s = o->shape();
                    if (s[0] == 1 xor s[1] == 1) // a face, not a corner
                        cut += static_cast<std::size_t>(std::max(s[0], s[1]));
                }
            }
        return cut;
    }
} // namespace


class HilbertCurve : public ::testing::TestWithParam<unsigned>
{
};

TEST_P(HilbertCurve, visitsEveryPointOnceThroughFaceNeighbours2D)
{
    auto const points = by_index<2>(GetParam());
    for (std::size_t k = 1; k < points.size(); ++k)
    {
        int steps = 0;
        for (std::size_t i = 0; i < 2; ++i)
            steps += std::abs(static_cast<int>(points[k][i]) - static_cast<int>(points[k - 1][i]));
        EXPECT_EQ(steps, 1) << "between " << k - 1 << " and " << k;
    }
}

TEST_P(HilbertCurve, visitsEveryPointOnceThroughFaceNeighbours3D)
{
    auto const points = by_index<3>(GetParam());
    for (std::size_t k = 1; k < points.size(); ++k)
    {
        int steps = 0;
        for (std::size_t i = 0; i < 3; ++i)
            steps += std::abs(static_cast<int>(points[k][i]) - static_cast<int>(points[k - 1][i]));
        EXPECT_EQ(steps, 1) << "between " << k - 1 << " and " << k;
    }
}

INSTANTIATE_TEST_SUITE_P(Partition, HilbertCurve, ::testing::Values(1u, 2u, 3u, 4u));


TEST(Partition, contiguousRunsWithinOneBoxOfTheShare)
{
    auto const boxes    = chop(Box<2>{std::array{0, 0}, std::array{99, 69}}, std::array{8, 8});
    auto const w        = cell_weights(boxes);
    auto const total    = std::accumulate(w.begin(), w.end(), 0.);
    auto const heaviest = *std::max_element(w.begin(), w.end());

    for (int const n_ranks : {1, 2, 3, 5, 7, 16})
    {
        auto const ranks = partition(boxes, w, n_ranks);
        ASSERT_EQ(ranks.size(), boxes.size());

        // along the curve the ranks only go up: contiguous runs
        auto const order = curve_order(boxes);
        for (std::size_t k = 1; k < order.size(); ++k)
            EXPECT_LE(ranks[order[k - 1]], ranks[order[k]]);

        for (auto const per_rank : rank_weights(ranks, w, n_ranks))
            EXPECT_LE(std::abs(per_rank - total / n_ranks), heaviest) << n_ranks << " ranks";

        EXPECT_EQ(partition(boxes, w, n_ranks), ranks); // deterministic
    }
}

TEST(Partition, moreRanksThanBoxesLeavesSomeEmpty)
{
    auto const boxes = chop(Box<2>{std::array{0, 0}, std::array{7, 7}}, std::array{4, 4});
    auto const ranks = partition(boxes, cell_weights(boxes), 8);
    std::set<int> const used(ranks.begin(), ranks.end());
    EXPECT_EQ(used.size(), boxes.size());
    for (auto const r : ranks)
        EXPECT_LT(r, 8);
}

TEST(Partition, keepsNeighboursTogetherBetterThanStrips)
{
    // 16 x 16 boxes of 4 cells, 4 ranks: quadrants along the curve, rows in strips
    auto const boxes = chop(Box<2>{std::array{0, 0}, std::array{63, 63}}, std::array{4, 4});
    std::vector<int> strips(boxes.size());
    for (std::size_t k = 0; k < boxes.size(); ++k)
        strips[k] = static_cast<int>(k * 4 / boxes.size());

    auto const curve = partition(boxes, cell_weights(boxes), 4);
    EXPECT_LT(cut_faces(boxes, curve), cut_faces(boxes, strips));
}

TEST(Partition, followsTheWeights)
{
    // PHARE's tests/simulator/test_load_balancing.py: two populations, one with no particles in
    // the cells of a row (density below the cutoff): by cells alone the ranks holding that row
    // have fewer particles
    int constexpr nx = 100, ny = 100, ppc_high = 90, ppc_low = 120, empty_row = 50;
    auto const boxes
        = chop(Box<2>{std::array{0, 0}, std::array{nx - 1, ny - 1}}, std::array{10, 5});

    std::vector<double> particles;
    for (auto const& b : boxes)
    {
        double n = 0;
        core::for_each_point(
            b, [&](auto const& p) { n += ppc_high + (p[1] == empty_row ? 0 : ppc_low); });
        particles.push_back(n);
    }
    auto const total = std::accumulate(particles.begin(), particles.end(), 0.);

    auto const spread = [&](std::vector<int> const& ranks, int const n_ranks) {
        auto const per_rank = rank_weights(ranks, particles, n_ranks);
        auto const [lo, hi] = std::minmax_element(per_rank.begin(), per_rank.end());
        return *hi - *lo;
    };

    int constexpr n_ranks   = 4;
    auto const by_cells     = partition(boxes, cell_weights(boxes), n_ranks);
    auto const by_particles = partition(boxes, particles, n_ranks);
    EXPECT_GT(spread(by_cells, n_ranks), 0.);
    EXPECT_LE(spread(by_particles, n_ranks), spread(by_cells, n_ranks));
    for (auto const per_rank : rank_weights(by_particles, particles, n_ranks))
        EXPECT_LE(std::abs(per_rank - total / n_ranks),
                  *std::max_element(particles.begin(), particles.end()));
}

TEST(Partition, rejectsBadInput)
{
    auto const boxes = chop(Box<2>{std::array{0, 0}, std::array{7, 7}}, std::array{4, 4});
    EXPECT_THROW(std::ignore = partition(boxes, cell_weights(boxes), 0), std::runtime_error);
    EXPECT_THROW(std::ignore = partition(boxes, std::vector<double>(3, 1.), 2), std::runtime_error);
    EXPECT_THROW(std::ignore = partition(boxes, std::vector<double>(4, -1.), 2),
                 std::runtime_error);
}

} // namespace damn::amr


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
