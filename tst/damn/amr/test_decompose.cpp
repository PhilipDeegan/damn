#include "damn/amr/decompose.hpp"
#include "damn/amr/cost.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/partition.hpp"
#include "damn/amr/topology.hpp"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <set>
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

    Hierarchy<2> hierarchy(Box2 const& domain)
    {
        Hierarchy<2> h;
        h.domain     = domain;
        h.periodic   = {true, true};
        h.max_levels = 3;
        h.ratio      = 2;
        return h;
    }

    // a weight per box: its cells
    Weigh<2> by_cells()
    {
        return [](int, std::vector<Box2> const& boxes) { return cell_weights(boxes); };
    }

    // a weight per box: its cells, plus `extra` per cell of `hot` in it (a refined region)
    Weigh<2> with_hot(Box2 const& hot, double const extra)
    {
        return [=](int, std::vector<Box2> const& boxes) {
            std::vector<double> out;
            for (auto const& b : boxes)
            {
                auto const o = b * hot;
                out.push_back(static_cast<double>(b.size())
                              + (o ? extra * static_cast<double>(o->size()) : 0.));
            }
            return out;
        };
    }

    // the patches cover `cells` once
    void expect_cover(Level<2> const& level, std::vector<Box2> const& cells)
    {
        std::size_t total = 0, in_patches = 0;
        for (auto const& b : cells)
            total += b.size();
        for (std::size_t i = 0; i < level.patches.size(); ++i)
        {
            in_patches += level.patches[i].box.size();
            for (std::size_t j = i + 1; j < level.patches.size(); ++j)
                EXPECT_FALSE(level.patches[i].box * level.patches[j].box) << i << " " << j;
        }
        EXPECT_EQ(in_patches, total);
    }

    // every patch at most `most` by weigh, or one unit wide (unsplittable)
    void expect_light(Level<2> const& level, Weigh<2> const& weigh, double const most,
                      int const unit)
    {
        for (auto const& p : level.patches)
        {
            auto const w         = weigh(level.number, {p.box}).front();
            auto const sh        = p.box.shape();
            bool const unit_wide = sh[0] <= unit and sh[1] <= unit;
            EXPECT_TRUE(w <= most or unit_wide) << p.box << " " << w;
        }
    }

    std::vector<double> loads(Level<2> const& level, int const n_ranks)
    {
        std::vector<double> out(static_cast<std::size_t>(n_ranks), 0.);
        for (auto const& p : level.patches)
            out[static_cast<std::size_t>(p.rank)] += static_cast<double>(p.box.size());
        return out;
    }
} // namespace

TEST(Decompose, LevelsOnOneRankIsChopAndCurve)
{
    auto const h        = hierarchy(box(0, 0, 39, 19));
    auto const topology = single_node(1);
    Decompose<2> const in{topology, by_cells(), {10, 10}};
    auto const level = decompose_level(h, 0, {h.domain}, in);
    auto const boxes = chop_evenly(h.domain, std::array<int, 2>{10, 10});
    ASSERT_EQ(level.patches.size(), boxes.size());
    auto const ranks = partition(boxes, cell_weights(boxes), node_share_capacity(topology));
    for (std::size_t i = 0; i < boxes.size(); ++i)
    {
        EXPECT_EQ(level.patches[i].box, boxes[i]);
        EXPECT_EQ(level.patches[i].rank, ranks[i]);
    }
}

TEST(Decompose, LevelsCutsOnePatchForEveryRank)
{
    auto const h        = hierarchy(box(0, 0, 63, 31));
    auto const topology = single_node(3);
    Decompose<2> const in{topology, by_cells(), {64, 32}}; // the domain: one patch
    auto const level = decompose_level(h, 0, {h.domain}, in);
    EXPECT_GE(level.patches.size(), 3u);
    auto const load = loads(level, 3);
    auto const mean = std::accumulate(load.begin(), load.end(), 0.) / 3.;
    for (auto const l : load)
        EXPECT_LE(std::abs(l - mean), 64.) << l; // within a block of the share (8 x 8)
    std::size_t cells = 0;
    for (auto const& p : level.patches)
        cells += p.box.size();
    EXPECT_EQ(cells, h.domain.size());
}

TEST(Decompose, RefinedLevelsCutInWholeCoarseCells)
{
    auto h              = hierarchy(box(0, 0, 31, 31));
    auto const topology = single_node(3);
    Decompose<2> const in{topology, by_cells(), {32, 32}};
    auto const levels = decompose(h, 0, {{h.domain}, {box(10, 10, 49, 29)}}, in);
    ASSERT_EQ(levels.size(), 2u);
    std::set<int> ranks;
    for (auto const& p : levels[1].patches)
    {
        ranks.insert(p.rank);
        for (std::size_t d = 0; d < 2; ++d)
        {
            EXPECT_EQ(p.box.lower[d] % 2, 0);
            EXPECT_EQ(p.box.shape()[d] % 2, 0);
        }
    }
    EXPECT_EQ(ranks.size(), 3u);
}

TEST(Decompose, KeepWithTheReplacedLevelKeepsEveryOwner)
{
    auto h              = hierarchy(box(0, 0, 39, 39));
    auto const topology = single_node(4);
    Decompose<2> levels_in{topology, by_cells(), {10, 10}};
    auto const made = decompose(h, 0, {{h.domain}, {box(20, 20, 59, 59)}}, levels_in);
    for (auto const& level : made)
        set_level(h, level);

    Decompose<2> keep_in{topology, by_cells(), {10, 10}, "keep"};
    // both levels again, together: level 1 must still see its old self
    auto const again = decompose(h, 0, {{h.domain}, {box(20, 20, 59, 59)}}, keep_in);
    ASSERT_EQ(again.size(), 2u);
    for (std::size_t ln = 0; ln < 2; ++ln)
    {
        auto const& before = made[ln].patches;
        auto const& after  = again[ln].patches;
        // every cell keeps its owner
        for (auto const& p : after)
            for (auto const& q : before)
                if (p.box * q.box)
                {
                    EXPECT_EQ(p.rank, q.rank) << "level " << ln;
                }
    }
}

TEST(Decompose, PatchWeightHalvesTheHeavyPatchesOnly)
{
    auto const h        = hierarchy(box(0, 0, 39, 39));
    auto const topology = single_node(1);
    auto const weigh    = with_hot(box(0, 0, 9, 9), 99.); // 100 per hot cell
    Decompose<2> in{topology, weigh, {20, 20}};
    auto const plain = decompose_level(h, 0, {h.domain}, in);
    ASSERT_EQ(plain.patches.size(), 4u);
    in.patch_weight  = 1000.;
    auto const level = decompose_level(h, 0, {h.domain}, in);
    expect_cover(level, {h.domain});
    expect_light(level, weigh, in.patch_weight, 1);
    // the patches away from the hot one as they were
    std::size_t away = 0;
    for (auto const& p : plain.patches)
        if (!(p.box * box(0, 0, 9, 9)))
        {
            ++away;
            EXPECT_TRUE(std::any_of(level.patches.begin(), level.patches.end(), [&](auto const& q) {
                return q.box == p.box;
            })) << p.box;
        }
    EXPECT_EQ(away, 3u);
    EXPECT_GT(level.patches.size(), 4u + 10u); // 10000 hot over patches of at most 1000
}

TEST(Decompose, PatchWeightKeepLeavesTheHalvesOnTheOwner)
{
    auto h              = hierarchy(box(0, 0, 39, 39));
    auto const topology = single_node(2);
    auto const weigh    = with_hot(box(20, 20, 29, 29), 99.);
    Decompose<2> in{topology, by_cells(), {20, 20}, "levels", -1., 8, false};
    auto const before = decompose_level(h, 0, {h.domain}, in);
    set_level(h, before);

    in.weigh         = weigh;
    in.partition     = "keep";
    in.patch_weight  = 2000.;
    auto const after = decompose_level(h, 0, {h.domain}, in);
    expect_cover(after, {h.domain});
    expect_light(after, weigh, in.patch_weight, 1);
    EXPECT_GT(after.patches.size(), before.patches.size());
    for (auto const& p : after.patches)
        for (auto const& q : before.patches)
            if (p.box * q.box)
            {
                EXPECT_EQ(p.rank, q.rank) << p.box;
            }
}

TEST(Decompose, PatchWeightOnARefinedLevelInWholeCoarseCells)
{
    auto h              = hierarchy(box(0, 0, 31, 31)); // ratio 2
    auto const topology = single_node(1);
    auto const weigh    = with_hot(box(20, 20, 27, 27), 49.);
    Decompose<2> in{topology, weigh, {32, 32}};
    in.patch_weight   = 500.;
    auto const levels = decompose(h, 0, {{h.domain}, {box(10, 10, 49, 29)}}, in);
    ASSERT_EQ(levels.size(), 2u);
    expect_cover(levels[1], {box(10, 10, 49, 29)});
    expect_light(levels[1], weigh, in.patch_weight, 2);
    for (auto const& p : levels[1].patches)
        for (std::size_t d = 0; d < 2; ++d)
        {
            EXPECT_EQ(p.box.lower[d] % 2, 0) << p.box;
            EXPECT_EQ(p.box.shape()[d] % 2, 0) << p.box;
        }
}

TEST(Decompose, PatchWeightStopsAtOneUnit)
{
    auto const h        = hierarchy(box(0, 0, 7, 7));
    auto const topology = single_node(1);
    Decompose<2> in{topology, by_cells(), {8, 8}};
    in.patch_weight  = 0.5; // below a cell: every patch halved to one cell
    auto const level = decompose_level(h, 0, {h.domain}, in);
    expect_cover(level, {h.domain});
    EXPECT_EQ(level.patches.size(), 64u);
}

TEST(Decompose, UnknownPartitionThrows)
{
    auto const h        = hierarchy(box(0, 0, 9, 9));
    auto const topology = single_node(1);
    Decompose<2> const in{topology, by_cells(), {10, 10}, "nope"};
    EXPECT_THROW((void)decompose_level(h, 0, {h.domain}, in), std::runtime_error);
}

TEST(Decompose, FinestRegridOfAFinerLevelFollowsTheFixedLevelUnderIt)
{
    // level 0 fixed (made by "levels" on 4 ranks); level 1 regridded alone, by decompose from 1
    //  and by decompose_level: every new patch with the owner of most coarse cells under it
    auto h              = hierarchy(box(0, 0, 63, 63));
    auto const topology = single_node(4);
    Decompose<2> levels{topology, by_cells(), {8, 8}};
    set_level(h, decompose_level(h, 0, {h.domain}, levels));
    Decompose<2> finest{topology, by_cells(), {8, 8}, "finest"};
    auto const fine = box(20, 20, 99, 99);
    for (auto const& level :
         {decompose(h, 1, {{fine}}, finest).front(), decompose_level(h, 1, {fine}, finest)})
    {
        ASSERT_FALSE(level.patches.empty());
        for (auto const& f : level.patches)
        {
            std::vector<double> cells(4, 0.);
            for (auto const& c : h.levels[0].patches)
                if (auto const o = core::coarsen(f.box, 2) * c.box)
                    cells[static_cast<std::size_t>(c.rank)] += static_cast<double>(o->size());
            EXPECT_EQ(f.rank,
                      std::distance(cells.begin(), std::max_element(cells.begin(), cells.end())));
        }
    }
}

TEST(Decompose, MinusIsDisjointAndOrdered)
{
    auto const pieces = core::minus_ordered(box(0, 0, 9, 9), box(3, 4, 5, 6));
    // x below, x above, then y below, y above narrowed to x 3..5
    std::vector<Box2> const expected{box(0, 0, 2, 9), box(6, 0, 9, 9), box(3, 0, 5, 3),
                                     box(3, 7, 5, 9)};
    EXPECT_EQ(pieces, expected);
    EXPECT_EQ(core::minus_ordered(box(0, 0, 1, 1), box(5, 5, 6, 6)),
              std::vector<Box2>{box(0, 0, 1, 1)});
}

TEST(Decompose, SplitUnderCoversTheBoxesOnce)
{
    auto const [under, rest]
        = detail::split_under<2>({box(0, 0, 19, 19)}, {box(10, 10, 29, 29)}, 2);
    ASSERT_EQ(under.size(), 1u);
    EXPECT_EQ(under[0], box(5, 5, 14, 14));
    std::size_t cells = under[0].size();
    for (auto const& b : rest)
    {
        cells += b.size();
        EXPECT_FALSE(b * under[0]);
    }
    EXPECT_EQ(cells, 400u);
}

TEST(Decompose, UnderSizeChopsTheCellsUnderTheFinerLevel)
{
    auto const h        = hierarchy(box(0, 0, 39, 39));
    auto const topology = single_node(1);
    Decompose<2> in{topology, by_cells(), {20, 20}};
    in.under_size     = {{5, 5}, {0, 0}};
    auto const levels = decompose(h, 0, {{h.domain}, {box(20, 20, 39, 39)}}, in);
    // the 10 x 10 coarse cells under level 1: 4 patches of 5 x 5 first
    for (std::size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(levels[0].patches[i].box.shape()[0], 5);
        EXPECT_TRUE(core::coarsen(box(20, 20, 39, 39), 2) * levels[0].patches[i].box);
    }
    std::size_t cells = 0;
    for (auto const& p : levels[0].patches)
        cells += p.box.size();
    EXPECT_EQ(cells, h.domain.size());
}

TEST(Decompose, FinestPutsCoarsePatchesWithTheFineAboveThem)
{
    auto const h        = hierarchy(box(0, 0, 63, 63));
    auto const topology = single_node(4);
    Decompose<2> in{topology, by_cells(), {8, 8}, "finest"};
    auto const fine   = box(32, 32, 95, 95); // over coarse cells 16..47
    auto const levels = decompose(h, 0, {{h.domain}, {fine}}, in);
    ASSERT_EQ(levels.size(), 2u);
    std::set<int> fine_ranks;
    for (auto const& p : levels[1].patches)
        fine_ranks.insert(p.rank);
    EXPECT_EQ(fine_ranks.size(), 4u);
    // no spill: every coarse patch under the fine level on the owner of the fine cells above it
    for (auto const& c : levels[0].patches)
    {
        std::vector<double> cells(4, 0.);
        for (auto const& f : levels[1].patches)
            if (auto const o = core::refine(c.box, 2) * f.box)
                cells[static_cast<std::size_t>(f.rank)] += static_cast<double>(o->size());
        if (std::accumulate(cells.begin(), cells.end(), 0.) > 0)
        {
            EXPECT_EQ(c.rank,
                      std::distance(cells.begin(), std::max_element(cells.begin(), cells.end())));
        }
    }
    // a level alone: level 0 as "levels"
    auto as_levels      = in;
    as_levels.partition = "levels";
    EXPECT_EQ(decompose_level(h, 0, {h.domain}, in).patches.size(),
              decompose_level(h, 0, {h.domain}, as_levels).patches.size());
}

TEST(Decompose, FinestInPiecesLeavesNoLevelOnOneRank)
{
    // the largest patch the domain (one patch a level, chopped): level 0 cut so every rank
    //  has work, the cells under level 1 with the owner of the fine cells above them
    auto const h        = hierarchy(box(0, 0, 63, 63));
    auto const topology = single_node(4);
    Decompose<2> in{topology, by_cells(), {128, 128}, "finest"};
    auto const levels = decompose(h, 0, {{h.domain}, {box(16, 16, 79, 79)}}, in);
    ASSERT_EQ(levels.size(), 2u);
    for (auto const& level : levels)
    {
        auto const load = loads(level, 4);
        for (auto const l : load)
            EXPECT_GT(l, 0.) << "level " << level.number;
    }
    for (auto const& c : levels[0].patches)
        for (auto const& f : levels[1].patches)
            if (core::coarsen(f.box, 2) * c.box)
            {
                EXPECT_EQ(c.rank, f.rank) << c.box << " under " << f.box;
            }
    // level 0 within a block row of its share (8 x 64 cells of 4096)
    auto const load0 = loads(levels[0], 4);
    EXPECT_LT(*std::max_element(load0.begin(), load0.end()), 4096. / 4 + 8 * 64);
}

TEST(Decompose, FinestKeepsASmallRefinedLevelOnFewRanks)
{
    // a refined level of one block (8 x 8 fine cells) on 4 ranks: one patch, not shredded
    auto const h        = hierarchy(box(0, 0, 63, 63));
    auto const topology = single_node(4);
    Decompose<2> in{topology, by_cells(), {128, 128}, "finest"};
    auto const levels = decompose(h, 0, {{h.domain}, {box(16, 16, 23, 23)}}, in);
    ASSERT_EQ(levels[1].patches.size(), 1u);
    for (std::size_t d = 0; d < 2; ++d)
        EXPECT_EQ(levels[1].patches[0].box.shape()[d] % 2, 0);
}

TEST(Decompose, FinestInPiecesThreeLevelsInWholeCoarseCells)
{
    // level 2 over odd level 1 cells (coarsened: 13..30 is level 1's 6..15, not whole level 0
    //  cells): level 1's pieces rounded out, every refined patch in whole coarser cells
    auto const h        = hierarchy(box(0, 0, 31, 31));
    auto const topology = single_node(3);
    Decompose<2> in{topology, by_cells(), {16, 16}, "finest"};
    auto const levels
        = decompose(h, 0, {{h.domain}, {box(8, 8, 47, 47)}, {box(26, 26, 61, 61)}}, in);
    ASSERT_EQ(levels.size(), 3u);
    // every level's cells once (no overlapping pieces)
    std::array<std::size_t, 3> const cells{32 * 32, 40 * 40, 36 * 36};
    for (std::size_t ln = 0; ln < 3; ++ln)
    {
        std::size_t total = 0;
        for (auto const& p : levels[ln].patches)
            total += p.box.size();
        EXPECT_EQ(total, cells[ln]) << "level " << ln;
    }
    for (std::size_t ln = 1; ln < 3; ++ln)
        for (auto const& p : levels[ln].patches)
            for (std::size_t d = 0; d < 2; ++d)
            {
                EXPECT_EQ(p.box.lower[d] % 2, 0) << "level " << ln << " " << p.box;
                EXPECT_EQ(p.box.shape()[d] % 2, 0) << "level " << ln << " " << p.box;
            }
}

TEST(Decompose, NearBandGoesToTheFinerOwner)
{
    // level 1 one block (8 x 8 fine cells, over coarse 8..11) on one rank: the level 0 cells
    //  within 2 of it (6..13) with that rank, every level 0 rank with work
    auto const h        = hierarchy(box(0, 0, 63, 63));
    auto const topology = single_node(4);
    Decompose<2> in{topology, by_cells(), {128, 128}, "finest"};
    in.near_band      = 2;
    auto const levels = decompose(h, 0, {{h.domain}, {box(16, 16, 23, 23)}}, in);
    ASSERT_EQ(levels[1].patches.size(), 1u);
    auto const owner = levels[1].patches[0].rank;
    for (auto const& p : levels[0].patches)
        if (p.box * box(6, 6, 13, 13))
        {
            EXPECT_EQ(p.rank, owner) << p.box;
        }
    for (auto const l : loads(levels[0], 4))
        EXPECT_GT(l, 0.);
}

TEST(Decompose, AlignedPutsFinePatchesOnTheCoarseUnderThem)
{
    auto const h        = hierarchy(box(0, 0, 63, 63));
    auto const topology = single_node(4);
    Decompose<2> in{topology, by_cells(), {8, 8}, "aligned"};
    auto const levels = decompose(h, 0, {{h.domain}, {box(32, 32, 95, 95)}}, in);
    for (auto const& f : levels[1].patches)
    {
        std::vector<double> cells(4, 0.);
        for (auto const& c : levels[0].patches)
            if (auto const o = core::coarsen(f.box, 2) * c.box)
                cells[static_cast<std::size_t>(c.rank)] += static_cast<double>(o->size());
        EXPECT_EQ(f.rank,
                  std::distance(cells.begin(), std::max_element(cells.begin(), cells.end())));
    }
}

TEST(Cost, ModelTerms)
{
    CostModel const m{2., 3., 5., 7., 1};
    // 4 x 2 cells, grown by 1: 6 x 4 = 24, 16 ghost cells
    EXPECT_DOUBLE_EQ(m(box(0, 0, 3, 1), 10.), 2. * 10 + 3. * 8 + 5. * 16 + 7.);
    EXPECT_DOUBLE_EQ(CostModel{}(box(0, 0, 3, 1), 10.), 10.); // the count alone
}

TEST(Cost, MapProratesBlocks)
{
    // 10 x 4 cells in blocks of 4 x 4: blocks of 4, 4 and 2 cells wide (the last clipped)
    CostMap<2> const map{box(0, 0, 9, 3), {4, 4}, {16., 32., 8.}};
    EXPECT_DOUBLE_EQ(map(box(0, 0, 9, 3)), 56.);
    EXPECT_DOUBLE_EQ(map(box(0, 0, 1, 3)), 8.);      // half the first block
    EXPECT_DOUBLE_EQ(map(box(3, 0, 4, 1)), 2. + 4.); // a column of 2 in each of two blocks
    EXPECT_DOUBLE_EQ(map(box(8, 0, 9, 3)), 8.);      // the clipped block, whole
    EXPECT_DOUBLE_EQ(map(box(20, 20, 21, 21)), 0.);  // outside
    EXPECT_THROW((void)(CostMap<2>{box(0, 0, 9, 3), {4, 4}, {1.}}(box(0, 0, 1, 1))),
                 std::runtime_error);
}

TEST(Cost, WeighByModelOverAMap)
{
    CostMap<2> const map{box(0, 0, 7, 7), {8, 8}, {64.}};
    auto const weigh = weigh_by<2>(CostModel{1., 1.}, counts_of<2>({map}));
    auto const w     = weigh(0, {box(0, 0, 3, 3), box(4, 0, 7, 7)});
    ASSERT_EQ(w.size(), 2u);
    EXPECT_DOUBLE_EQ(w[0], 16. + 16.);
    EXPECT_DOUBLE_EQ(w[1], 32. + 32.);
    EXPECT_THROW((void)weigh(1, {box(0, 0, 1, 1)}), std::runtime_error);
}

} // namespace damn::amr

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
