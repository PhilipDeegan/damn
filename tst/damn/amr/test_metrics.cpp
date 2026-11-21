#include "damn/amr/metrics.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/topology.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <vector>

using namespace damn;
using namespace damn::amr;

namespace
{
core::Box<int, 1> box1(int const lower, int const upper)
{
    return {core::Point<int, 1>{lower}, core::Point<int, 1>{upper}};
}

std::size_t at(std::array<std::size_t, n_distances> const& a, Distance const d)
{
    return a[static_cast<std::size_t>(d)];
}

// 40 periodic cells: level 0 four patches of 10 (ranks 0 0 1 1, one node), level 1 the refined
// cells 10..19 (fine 20..39) on rank 0
Hierarchy<1> two_levels()
{
    Hierarchy<1> h;
    h.domain     = box1(0, 39);
    h.dx         = {1.};
    h.periodic   = {true};
    h.max_levels = 2;
    h.ratio      = 2;
    using Boxes  = std::vector<core::Box<int, 1>>;
    h.levels.push_back(make_level(0, Boxes{box1(0, 9), box1(10, 19), box1(20, 29), box1(30, 39)},
                                  std::vector<int>{0, 0, 1, 1}));
    h.levels.push_back(make_level(1, Boxes{box1(20, 39)}, std::vector<int>{0}));
    return h;
}
} // namespace

TEST(Metrics, countTheCellsExchangedByDistance)
{
    auto const m = metrics(two_levels(), single_node(2), /*reach=*/2);
    ASSERT_EQ(m.size(), 2u);

    // level 0: each patch reads 2 cells of each neighbour, the domain's ends through the
    // periodic image: 0<->1 and 2<->3 on one rank, 1<->2 and 3<->0 across ranks of a node
    EXPECT_EQ(m[0].patches, 4u);
    EXPECT_EQ(m[0].cells, 40u);
    EXPECT_EQ(m[0].smallest_patch, 10u);
    EXPECT_EQ(at(m[0].ghost_cells, Distance::Local), 8u);
    EXPECT_EQ(at(m[0].ghost_cells, Distance::Node), 8u);
    EXPECT_EQ(at(m[0].ghost_cells, Distance::Self), 0u);
    EXPECT_EQ(m[0].weight, (std::vector<double>{20., 20.}));
    EXPECT_DOUBLE_EQ(m[0].imbalance(), 1.);

    // level 1: its box grown by 2 (18..41) is coarse 9..20: cell 9 and 10..19 on rank 0, cell 20
    // on rank 1; no other fine patch, its periodic images too far
    EXPECT_EQ(at(m[1].coarse_cells, Distance::Local), 11u);
    EXPECT_EQ(at(m[1].coarse_cells, Distance::Node), 1u);
    EXPECT_EQ(at(m[1].ghost_cells, Distance::Self), 0u);
    EXPECT_EQ(m[1].weight, (std::vector<double>{20., 0.}));
    EXPECT_DOUBLE_EQ(m[1].imbalance(), 2.);
}

TEST(Metrics, weightsPerPatchAndRemoteRanks)
{
    auto t          = single_node(2);
    t.ranks[1].node = 1; // rank 1 on another node
    auto const m    = metrics(two_levels(), t, 2, {{1., 1., 3., 3.}});
    EXPECT_EQ(m[0].weight, (std::vector<double>{2., 6.}));
    EXPECT_DOUBLE_EQ(m[0].imbalance(), 1.5);
    EXPECT_EQ(at(m[0].ghost_cells, Distance::Remote), 8u);
    EXPECT_EQ(at(m[1].coarse_cells, Distance::Remote), 1u);
    EXPECT_EQ(m[1].weight, (std::vector<double>{20., 0.})); // no weights given: cells
}

TEST(Metrics, anotherMemoryDomainOfTheNodeIsASocketAway)
{
    // ranks 0 and 1 on one node: in one memory domain Node, in two Socket (a dual socket node)
    auto t = single_node(2);
    EXPECT_EQ(distance(t, 0, 1), Distance::Node);
    t.ranks[1].numa = 1;
    EXPECT_EQ(distance(t, 0, 1), Distance::Socket);
    EXPECT_LT(Distance::Node, Distance::Socket);
    EXPECT_LT(Distance::Socket, Distance::Remote);
    auto const m = metrics(two_levels(), t, /*reach=*/2);
    EXPECT_EQ(at(m[0].ghost_cells, Distance::Socket), 8u); // the Node cells of the same layout
    EXPECT_EQ(at(m[0].ghost_cells, Distance::Node), 0u);
    t.ranks[1].node = 1;
    EXPECT_EQ(distance(t, 0, 1), Distance::Remote);
}

TEST(Metrics, migrationCountsMovedAndRefinedCellsByDistance)
{
    using Boxes    = std::vector<core::Box<int, 1>>;
    auto from      = two_levels(); // level 0 ranks 0 0 1 1, level 1 fine 20..39 on rank 0
    from.levels[0] = make_level(0, Boxes{box1(0, 19), box1(20, 39)}, std::vector<int>{0, 1});

    auto to      = two_levels();
    to.levels[0] = make_level(0, Boxes{box1(0, 19), box1(20, 39)}, std::vector<int>{1, 0});
    to.levels[1] = make_level(1, Boxes{box1(20, 59)}, std::vector<int>{1}); // coarse 10..29

    auto const m = migration(from, to, single_node(2));
    ASSERT_EQ(m.size(), 2u);
    // level 0: both halves change rank
    EXPECT_EQ(at(m[0].moved, Distance::Node), 40u);
    EXPECT_EQ(at(m[0].moved, Distance::Local), 0u);
    // level 1: fine 20..39 held on rank 0, now rank 1; fine 40..59 new, refined from the old
    //  coarse 20..29 of rank 1: stays on its rank
    EXPECT_EQ(at(m[1].moved, Distance::Node), 20u);
    EXPECT_EQ(at(m[1].refined, Distance::Local), 20u);
    EXPECT_EQ(at(m[1].refined, Distance::Node), 0u);

    // nothing changes: nothing moves off its rank
    auto const same = migration(from, from, single_node(2));
    for (auto const& lm : same)
    {
        EXPECT_EQ(at(lm.moved, Distance::Node), 0u);
        EXPECT_EQ(at(lm.refined, Distance::Node) + at(lm.refined, Distance::Local), 0u);
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
