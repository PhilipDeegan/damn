#include "damn/amr/keep_partition.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/topology.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"

#include "gtest/gtest.h"

#include <cstddef>
#include <map>
#include <numeric>
#include <stdexcept>
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

    // each cell's owner, every cell once
    std::map<std::pair<int, int>, int> owners(std::vector<OwnedBox<2>> const& pieces)
    {
        std::map<std::pair<int, int>, int> out;
        for (auto const& p : pieces)
            core::for_each_point(p.box, [&](auto const& c) {
                auto const [_, fresh] = out.emplace(std::pair{c[0], c[1]}, p.rank);
                EXPECT_TRUE(fresh) << "a cell in two pieces";
            });
        return out;
    }

    Topology nodes(std::vector<int> const& node_of_rank)
    {
        auto t = single_node(static_cast<int>(node_of_rank.size()));
        for (std::size_t r = 0; r < node_of_rank.size(); ++r)
            t.ranks[r].node = node_of_rank[r];
        return t;
    }
} // namespace

TEST(Keep, oldOwnersThenCoarserOwnersThenFresh)
{
    // the new level: cells 0..19 x 0..9; the old level held 0..9 x 0..9 (rank 1); the coarser
    //  level (ratio 2) has 5..7 x 0..4 on rank 2 (fine 10..15 x 0..9); fine 16..19 is fresh
    std::vector<Patch<2>> const old{{box(0, 0, 9, 9), 1, 0}};
    std::vector<Patch<2>> const coarser{{box(5, 0, 7, 4), 2, 0}};
    std::vector<Box2> fresh_boxes;
    auto const pieces = keep<2>({box(0, 0, 19, 9)}, old, coarser, 2, [&](auto const& rest) {
        fresh_boxes = rest;
        std::vector<OwnedBox<2>> out;
        for (auto const& b : rest)
            out.push_back({b, 3});
        return out;
    });
    auto const own    = owners(pieces);
    EXPECT_EQ(own.size(), 200u); // every cell once
    for (auto const& [c, r] : own)
        EXPECT_EQ(r, c.first < 10 ? 1 : c.first < 16 ? 2 : 3) << c.first << "," << c.second;
    std::size_t fresh_cells = 0;
    for (auto const& b : fresh_boxes)
        fresh_cells += b.size();
    EXPECT_EQ(fresh_cells, 40u);
}

TEST(Keep, nothingOldOrCoarserIsAllFresh)
{
    std::size_t calls = 0;
    auto const pieces
        = keep<2>({box(0, 0, 3, 3), box(10, 10, 11, 11)}, {}, {}, 2, [&](auto const& rest) {
              ++calls;
              std::vector<OwnedBox<2>> out;
              for (auto const& b : rest)
                  out.push_back({b, 0});
              return out;
          });
    EXPECT_EQ(calls, 1u);
    EXPECT_EQ(owners(pieces).size(), 20u);
}

TEST(Keep, freshMustCoverTheRest)
{
    EXPECT_THROW((void)keep<2>({box(0, 0, 3, 3)}, {}, {}, 2,
                               [](auto const&) { return std::vector<OwnedBox<2>>{}; }),
                 std::runtime_error);
}

TEST(OwnedBoxes, coalescedPerRank)
{
    std::vector<OwnedBox<2>> const pieces{
        {box(0, 0, 4, 9), 0}, {box(5, 0, 9, 9), 0}, {box(10, 0, 14, 9), 1}};
    auto const per_rank = owned_boxes(pieces, 3);
    ASSERT_EQ(per_rank.size(), 3u);
    ASSERT_EQ(per_rank[0].size(), 1u);
    EXPECT_EQ(per_rank[0][0], box(0, 0, 9, 9));
    EXPECT_EQ(per_rank[1].size(), 1u);
    EXPECT_TRUE(per_rank[2].empty());
    EXPECT_THROW((void)owned_boxes(pieces, 1), std::runtime_error);
}

TEST(Spill, withinTheToleranceOfTheShare)
{
    // 8 patches of weight 1 all on rank 0 of 4 (one node): 2 each after
    std::vector<Box2> boxes;
    for (int i = 0; i < 8; ++i)
        boxes.push_back(box(i, 0, i, 0));
    auto const ranks
        = spill(boxes, std::vector<int>(8, 0), std::vector<double>(8, 1.), single_node(4), 0.1);
    std::vector<int> count(4, 0);
    for (auto const r : ranks)
        ++count[static_cast<std::size_t>(r)];
    EXPECT_EQ(count, (std::vector<int>{2, 2, 2, 2}));
}

TEST(Spill, itsNodeFirst)
{
    // ranks 0, 1 on node 0, ranks 2, 3 on node 1; rank 0 holds 3 of 4 weight units, rank 2 one;
    //  a share of 1, tol 1: at most 2 a rank. Rank 0's excess goes to rank 1 (same node), not
    //  to the emptier rank 3
    std::vector<Box2> const boxes{box(0, 0, 0, 0), box(1, 0, 1, 0), box(2, 0, 2, 0),
                                  box(3, 0, 3, 0)};
    auto const ranks
        = spill(boxes, {0, 0, 0, 2}, std::vector<double>(4, 1.), nodes({0, 0, 1, 1}), 1.);
    std::vector<int> count(4, 0);
    for (auto const r : ranks)
        ++count[static_cast<std::size_t>(r)];
    EXPECT_EQ(count, (std::vector<int>{2, 1, 1, 0}));
}

TEST(Spill, itsMemoryDomainBeforeItsNode)
{
    // one node, two memory domains: ranks 0, 1 in domain 0, ranks 2, 3 in domain 1; rank 0's
    //  excess goes to rank 1 (its domain), not to the emptier rank 3 (the other socket)
    std::vector<Box2> const boxes{box(0, 0, 0, 0), box(1, 0, 1, 0), box(2, 0, 2, 0),
                                  box(3, 0, 3, 0)};
    auto t          = single_node(4);
    t.ranks[2].numa = t.ranks[3].numa = 1;
    auto const ranks = spill(boxes, {0, 0, 0, 2}, std::vector<double>(4, 1.), t, 1.);
    std::vector<int> count(4, 0);
    for (auto const r : ranks)
        ++count[static_cast<std::size_t>(r)];
    EXPECT_EQ(count, (std::vector<int>{2, 1, 1, 0}));
}

TEST(Spill, keepsARanksLastPatch)
{
    // one heavy patch over the share: nowhere to split it
    auto const ranks
        = spill<2>({box(0, 0, 9, 9), box(10, 0, 10, 0)}, {0, 1}, {100., 1.}, single_node(2), 0.1);
    EXPECT_EQ(ranks, (std::vector<int>{0, 1}));
}

TEST(Spill, rejectsBadInput)
{
    EXPECT_THROW((void)spill<2>({box(0, 0, 1, 1)}, {}, {1.}, single_node(2), 0.1),
                 std::runtime_error);
}

} // namespace damn::amr

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
