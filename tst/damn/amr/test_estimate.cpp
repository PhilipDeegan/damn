// damn::amr::estimate: one rank exchanges for free and its step is its compute times each
// level's substeps; two ranks of a node pay the Node distance's latency and cells for the
// overlap within reach, the same on two nodes Remote's; threads share a rank's tiles;
// exchange_costs is relative to Node

#include "damn/amr/estimate.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/topology.hpp"

#include "gtest/gtest.h"

#include <cstddef>
#include <vector>

namespace damn::amr
{
namespace
{
    using Box2 = core::Box<int, 2>;

    Box2 box(int x0, int y0, int x1, int y1)
    {
        return Box2{{x0, y0}, {x1, y1}};
    }

    // two 10 x 10 patches side by side, not periodic: each sees 1 x 10 of the other at reach 1
    Hierarchy<2> pair(int const rank_b)
    {
        Hierarchy<2> h;
        h.domain     = box(0, 0, 19, 9);
        h.periodic   = {false, false};
        h.max_levels = 2;
        h.ratio      = 2;
        h.levels.push_back(make_level<2>(0, {box(0, 0, 9, 9), box(10, 0, 19, 9)}, {0, rank_b}));
        return h;
    }

    Topology two_nodes()
    {
        auto t          = single_node(2);
        t.ranks[1].node = 1;
        return t;
    }
} // namespace

TEST(Estimate, OneRankExchangesForFree)
{
    auto h = pair(0);
    h.levels.push_back(make_level<2>(1, {box(10, 0, 29, 19)}, {0}));
    auto const c = exchange_costs(1e-6, 1e-9, 10.);
    auto const e = estimate(h, single_node(1), 1, {{3., 5.}, {7.}}, c);
    ASSERT_EQ(e.levels.size(), 2u);
    EXPECT_DOUBLE_EQ(e.levels[0].exchange[0], 0.);
    EXPECT_DOUBLE_EQ(e.levels[1].exchange[0], 0.);
    EXPECT_DOUBLE_EQ(e.step, (3. + 5.) + 2. * 7.); // level 1: 2 substeps
}

TEST(Estimate, NeighboursPayTheirDistance)
{
    auto const c   = exchange_costs(1e-6, 1e-9, 10.); // Node: 1 us, 1e-8 s a cell
    auto const one = estimate(pair(1), single_node(2), 1, {{1., 1.}}, c);
    // each rank: one message, 10 cells (a column of the other patch)
    EXPECT_DOUBLE_EQ(one.levels[0].exchange[0], 1e-6 + 10 * 1e-8);
    EXPECT_DOUBLE_EQ(one.levels[0].exchange[1], 1e-6 + 10 * 1e-8);
    EXPECT_DOUBLE_EQ(one.step, 1. + 1e-6 + 10 * 1e-8);
    auto const far = estimate(pair(1), two_nodes(), 1, {{1., 1.}}, c);
    EXPECT_DOUBLE_EQ(far.levels[0].exchange[0], 10. * (1e-6 + 10 * 1e-8)); // Remote: 10 x Node
}

TEST(Estimate, CoarserCellsUnderAFinePatchOnAnotherRank)
{
    auto h = pair(0);                                             // level 0 all on rank 0
    h.levels.push_back(make_level<2>(1, {box(0, 0, 3, 3)}, {1})); // over coarse 0..1, grown: 0..2
    auto const c = exchange_costs(1e-6, 1e-9, 10.);
    auto const e = estimate(h, single_node(2), 1, {}, c);
    // grown by 1: -1..4 fine, coarsened -1..2, clipped by the patches: 0..2 x 0..2, 9 cells
    EXPECT_DOUBLE_EQ(e.levels[1].exchange[1], 1e-6 + 9 * 1e-8);
    EXPECT_DOUBLE_EQ(e.levels[1].exchange[0], 0.);
}

TEST(Estimate, ThreadsShareARanksTiles)
{
    // one rank, two 10 x 10 patches of cost 4 and 2: one thread their sum; threads: no busier
    //  than the sum, at least the largest share, efficiency dividing; the main thread busy half
    //  its time with one thread: twice the sum
    auto const h = pair(0);
    auto const c = exchange_costs(0., 0., 0.);
    auto const t = single_node(1);
    EXPECT_DOUBLE_EQ(estimate(h, t, 1, {{4., 2.}}, c).step, 6.);
    auto const two = estimate(h, t, 1, {{4., 2.}}, c, Threads{.per_rank = 2}).step;
    EXPECT_LE(two, 6.);
    EXPECT_GE(two, 3.);
    EXPECT_DOUBLE_EQ(
        estimate(h, t, 1, {{4., 2.}}, c, Threads{.per_rank = 2, .efficiency = 0.5}).step,
        two / 0.5);
    EXPECT_DOUBLE_EQ(estimate(h, t, 1, {{4., 2.}}, c, Threads{.main_busy = 0.5}).step, 12.);
}

TEST(Estimate, ExchangeCostsAreRelativeToNode)
{
    auto const c = exchange_costs(2., 3., 4.);
    EXPECT_DOUBLE_EQ(c.latency[static_cast<std::size_t>(Distance::Node)], 2.);
    EXPECT_DOUBLE_EQ(c.per_cell[static_cast<std::size_t>(Distance::Node)], 12.);
    EXPECT_DOUBLE_EQ(c.latency[static_cast<std::size_t>(Distance::Remote)], 20.);
    EXPECT_DOUBLE_EQ(c.latency[static_cast<std::size_t>(Distance::Local)], 0.);
    // no bootstrap on this build: the defaults
    auto const b = bootstrapped_exchange_costs(8.);
    EXPECT_DOUBLE_EQ(b.latency[static_cast<std::size_t>(Distance::Node)],
                     bootstrap::number("mpi_exchange_node_latency_s", 1e-6));
}

} // namespace damn::amr

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
