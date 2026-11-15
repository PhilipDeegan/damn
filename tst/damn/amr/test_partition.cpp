// damn::amr::partition: the Hilbert curve visits every point once through face neighbours,
// partitions are contiguous runs of the curve within one box of the even share, keep neighbours
// together better than lexicographic strips, and follow the weights (PHARE's load balancing
// test: a row of cells with no particles of one population); cut_partition cuts the boxes a
// rank's share ends in on their blocks: the pieces cover each box once, and the ranks end within
// a block of the share where whole boxes do not

#include "damn/amr/partition.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/topology.hpp"
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
                    if ((s[0] == 1) != (s[1] == 1)) // a face, not a corner
                        cut += static_cast<std::size_t>(std::max(s[0], s[1]));
                }
            }
        return cut;
    }

    // pieces cover the boxes once: as many cells, no two overlap, each inside one box
    template<std::size_t dim>
    void expect_covers_once(std::vector<Box<dim>> const& pieces, std::vector<Box<dim>> const& boxes)
    {
        std::size_t cells = 0, piece_cells = 0;
        for (auto const& b : boxes)
            cells += b.size();
        for (std::size_t a = 0; a < pieces.size(); ++a)
        {
            piece_cells += pieces[a].size();
            EXPECT_TRUE(std::any_of(boxes.begin(), boxes.end(), [&](auto const& b) {
                auto const o = b * pieces[a];
                return o and o->size() == pieces[a].size();
            }));
            for (std::size_t b = a + 1; b < pieces.size(); ++b)
                EXPECT_FALSE(pieces[a] * pieces[b]);
        }
        EXPECT_EQ(piece_cells, cells);
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


TEST(Blocks, fromTheLowerCornerRemainderInTheLast)
{
    Box<2> const box{std::array{3, -5}, std::array{22, 4}}; // 20 x 10
    auto const bs = blocks_of(box, std::array{8, 8});
    ASSERT_EQ(bs.size(), 2u); // x: 8 + 12, y: 10 (less than two blocks)
    EXPECT_EQ(bs[0], (Box<2>{std::array{3, -5}, std::array{10, 4}}));
    EXPECT_EQ(bs[1], (Box<2>{std::array{11, -5}, std::array{22, 4}}));
    expect_covers_once(bs, {box});

    Box<2> const small{std::array{0, 0}, std::array{2, 2}};
    EXPECT_EQ(blocks_of(small, std::array{8, 8}), std::vector{small});
    EXPECT_EQ(coalesce(blocks_of(Box<2>{std::array{0, 0}, std::array{63, 40}}, std::array{8, 8})),
              (std::vector{Box<2>{std::array{0, 0}, std::array{63, 40}}}));
}

TEST(CutPartition, cutsTheBoxesASharesEndsInWithinABlock)
{
    // two large patches (tiled ones): whole boxes cannot balance 3 or more ranks
    std::vector const boxes{Box<2>{std::array{0, 0}, std::array{63, 63}},
                            Box<2>{std::array{64, 0}, std::array{127, 63}}};
    std::array const block{8, 8};
    auto const bs    = blocks(boxes, block);
    auto const w     = cell_weights(bs);
    auto const total = std::accumulate(w.begin(), w.end(), 0.);

    for (int const n_ranks : {1, 2, 3, 5, 7, 16})
    {
        auto const p = cut_partition(CutPartition{.boxes         = boxes,
                                                  .block         = block,
                                                  .block_weights = w,
                                                  .capacity      = equal_capacity(n_ranks),
                                                  .tol           = 0.});
        ASSERT_EQ(p.boxes.size(), p.ranks.size());
        expect_covers_once(p.boxes, boxes);
        EXPECT_TRUE(std::is_sorted(p.ranks.begin(), p.ranks.end()));

        std::vector<double> per_rank(n_ranks, 0);
        for (std::size_t k = 0; k < p.boxes.size(); ++k)
            per_rank[p.ranks[k]] += static_cast<double>(p.boxes[k].size());
        for (auto const r : per_rank)
            EXPECT_LE(std::abs(r - total / n_ranks), 64.) << n_ranks << " ranks"; // a block

        auto const again = cut_partition(CutPartition{.boxes         = boxes,
                                                      .block         = block,
                                                      .block_weights = w,
                                                      .capacity      = equal_capacity(n_ranks),
                                                      .tol           = 0.}); // deterministic
        EXPECT_EQ(again.boxes, p.boxes);
        EXPECT_EQ(again.ranks, p.ranks);
    }
}

TEST(CutPartition, followsTheBlockWeights)
{
    // the particles of PHARE's load balancing test, counted per block of one large patch
    std::vector const boxes{Box<2>{std::array{0, 0}, std::array{99, 99}}};
    std::array const block{10, 10};
    std::vector<double> w;
    for (auto const& b : blocks(boxes, block))
    {
        double n = 0;
        core::for_each_point(b, [&](auto const& c) { n += 90 + (c[1] == 50 ? 0 : 120); });
        w.push_back(n);
    }
    auto const total = std::accumulate(w.begin(), w.end(), 0.);
    auto const p     = cut_partition(CutPartition{
        .boxes = boxes, .block = block, .block_weights = w, .capacity = equal_capacity(4)});
    expect_covers_once(p.boxes, boxes);

    std::vector<double> per_rank(4, 0);
    for (std::size_t k = 0; k < p.boxes.size(); ++k)
        core::for_each_point(p.boxes[k], [&](auto const& c) {
            per_rank[p.ranks[k]] += 90 + (c[1] == 50 ? 0 : 120);
        });
    for (auto const r : per_rank)
        EXPECT_LE(std::abs(r - total / 4), *std::max_element(w.begin(), w.end()));
}

TEST(CutPartition, smallBoxesWithinTolAreNotCut)
{
    auto const boxes = chop(Box<2>{std::array{0, 0}, std::array{99, 69}}, std::array{8, 8});
    std::array const block{4, 4};
    auto const w = cell_weights(blocks(boxes, block));

    // a box is at most 64 cells, a share of 4 ranks 1750: tol 0.05 of it is more than half a box
    auto const p     = cut_partition(CutPartition{
        .boxes = boxes, .block = block, .block_weights = w, .capacity = equal_capacity(4)});
    auto const whole = partition(boxes, cell_weights(boxes), 4);
    auto const order = curve_order(boxes);
    ASSERT_EQ(p.boxes.size(), boxes.size());
    for (std::size_t k = 0; k < order.size(); ++k)
    {
        EXPECT_EQ(p.boxes[k], boxes[order[k]]);
        EXPECT_EQ(p.ranks[k], whole[order[k]]);
    }
}

TEST(CutPartition, followsTheCapacities)
{
    // rank 1 has half its threads on something else, rank 3 none for this: shares 2 : 1 : 2 : 0
    std::vector const boxes{Box<2>{std::array{0, 0}, std::array{63, 63}},
                            Box<2>{std::array{64, 0}, std::array{127, 63}}};
    std::array const block{8, 8};
    auto const w = cell_weights(blocks(boxes, block));
    std::vector const capacity{1., 0.5, 1., 0.};
    double const total = 128. * 64;

    auto const p = cut_partition(CutPartition{
        .boxes = boxes, .block = block, .block_weights = w, .capacity = capacity, .tol = 0.});
    expect_covers_once(p.boxes, boxes);
    EXPECT_TRUE(std::is_sorted(p.ranks.begin(), p.ranks.end()));
    std::vector<double> per_rank(capacity.size(), 0);
    for (std::size_t k = 0; k < p.boxes.size(); ++k)
        per_rank[p.ranks[k]] += static_cast<double>(p.boxes[k].size());
    for (std::size_t r = 0; r < capacity.size(); ++r)
        EXPECT_LE(std::abs(per_rank[r] - total * capacity[r] / 2.5), 64.) << r; // a block
    EXPECT_EQ(per_rank[3], 0.);

    // whole boxes, the same shares: within a box of each
    auto const small = chop(Box<2>{std::array{0, 0}, std::array{99, 69}}, std::array{8, 8});
    auto const sw    = cell_weights(small);
    auto const ranks = partition(small, sw, capacity);
    std::vector<double> whole(capacity.size(), 0);
    for (std::size_t k = 0; k < small.size(); ++k)
        whole[ranks[k]] += sw[k];
    for (std::size_t r = 0; r < capacity.size(); ++r)
        EXPECT_LE(std::abs(whole[r] - 7000. * capacity[r] / 2.5), 64.) << r;
    EXPECT_EQ(partition(small, sw, std::vector(4, 1.)), partition(small, sw, 4));
}

TEST(CutPartition, anEqualShareOfEachNode)
{
    // 2 nodes, 4 ranks on the first and 2 on the second: a rank of the second does twice the work
    Topology t = single_node(6);
    for (int r = 4; r < 6; ++r)
        t.ranks[r].node = 1;
    auto const capacity = node_share_capacity(t);
    EXPECT_EQ(capacity, (std::vector{0.25, 0.25, 0.25, 0.25, 0.5, 0.5}));
    EXPECT_EQ(node_share_capacity(single_node(3)), (std::vector{1. / 3, 1. / 3, 1. / 3}));

    std::vector const boxes{Box<2>{std::array{0, 0}, std::array{127, 63}}};
    std::array const block{8, 8};
    auto const p = cut_partition(CutPartition{.boxes         = boxes,
                                              .block         = block,
                                              .block_weights = cell_weights(blocks(boxes, block)),
                                              .capacity      = capacity,
                                              .tol           = 0.});
    std::vector<double> per_rank(capacity.size(), 0);
    for (std::size_t k = 0; k < p.boxes.size(); ++k)
        per_rank[p.ranks[k]] += static_cast<double>(p.boxes[k].size());
    for (std::size_t r = 0; r < capacity.size(); ++r)
        EXPECT_LE(std::abs(per_rank[r] - 128. * 64 * capacity[r] / 2), 64.) << r;
}

TEST(CutPartition, rejectsBadInput)
{
    std::vector const boxes{Box<2>{std::array{0, 0}, std::array{15, 15}}};
    std::array const block{8, 8};
    auto const w = cell_weights(blocks(boxes, block));
    EXPECT_THROW(
        std::ignore = cut_partition(CutPartition{
            .boxes = boxes, .block = block, .block_weights = w, .capacity = equal_capacity(0)}),
        std::runtime_error);
    EXPECT_THROW(std::ignore
                 = cut_partition(CutPartition{.boxes         = boxes,
                                              .block         = block,
                                              .block_weights = std::vector<double>(3, 1.),
                                              .capacity      = equal_capacity(2)}),
                 std::runtime_error);
    EXPECT_THROW(std::ignore
                 = cut_partition(CutPartition{.boxes         = boxes,
                                              .block         = block,
                                              .block_weights = std::vector<double>(4, -1.),
                                              .capacity      = equal_capacity(2)}),
                 std::runtime_error);
    EXPECT_THROW(std::ignore = cut_partition(CutPartition{.boxes         = boxes,
                                                          .block         = std::array{0, 8},
                                                          .block_weights = w,
                                                          .capacity      = equal_capacity(2)}),
                 std::runtime_error);
    for (auto const& capacity : {std::vector<double>{}, std::vector{1., -1.}, std::vector{0., 0.}})
    {
        EXPECT_THROW(std::ignore = cut_partition(CutPartition{
                         .boxes = boxes, .block = block, .block_weights = w, .capacity = capacity}),
                     std::runtime_error);
        EXPECT_THROW(std::ignore = partition(boxes, std::vector(1, 1.), capacity),
                     std::runtime_error);
    }
}

} // namespace damn::amr


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
