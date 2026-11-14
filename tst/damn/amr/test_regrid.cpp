// damn::amr::regrid on one rank: the toy level (8 x 6 cells, 3 x 4 patches) onto other
// decompositions of the same cells

#include "damn/amr/hierarchy.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/regrid.hpp"
#include "tst/damn/amr/toy.hpp"
#include "tst/damn/amr/toy_regrid.hpp"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

namespace damn::amr
{
namespace
{
    struct Regrid
    {
        Hierarchy<2> h = make_hierarchy({true, false});
        LevelData_t ld = allocate(h.levels[0], 0, make_patch_data);

        Regrid() { fill(h, ld); }

        auto onto(Level<2> const& level)
        {
            toy::Tracker::reset_peak();
            auto const map = make_regrid_map(h.levels[0], level, 0);
            ld             = regrid(map, std::move(ld), make_patch_data);
            return map;
        }
    };
} // namespace


TEST(Regrid, ontoSmallerPatches)
{
    Regrid r;
    auto const before = r.ld.data.size();
    auto const level  = make_level(0, chop(r.h.domain, {2, 2}));
    auto const map    = r.onto(level);

    ASSERT_EQ(r.ld.patches, level.patches);
    EXPECT_EQ(check(r.h, r.ld), 0u);
    EXPECT_EQ(domain_items(r.ld), 48u);
    // the 2 x 2 corner patch is the same box: kept, ghosts included. The others' ghosts are
    //  refilled after
    for (std::size_t i = 0; i < r.ld.data.size(); ++i)
        EXPECT_EQ(r.ld.data[i].patch_ghost.items.size(), map.kept[i] == not_local ? 0u : 1u);
    EXPECT_EQ(std::count_if(map.kept.begin(), map.kept.end(),
                            [](auto const k) { return k != not_local; }),
              1);

    // old patches freed as they are visited: never the old and the new level at once
    std::cout << "patch datas alive at most: " << toy::Tracker::peak << " (" << before << " old, "
              << level.patches.size() << " new)\n";
    EXPECT_LT(toy::Tracker::peak, static_cast<long>(before + level.patches.size()));
    EXPECT_EQ(toy::Tracker::live, static_cast<long>(level.patches.size()));
}

TEST(Regrid, ontoLargerPatches)
{
    Regrid r;
    auto const level = make_level(0, chop(r.h.domain, {8, 3}));
    r.onto(level);
    EXPECT_EQ(r.ld.data.size(), 2u);
    EXPECT_EQ(check(r.h, r.ld), 0u);
    EXPECT_EQ(domain_items(r.ld), 48u);
}

TEST(Regrid, sameBoxesKeepTheirData)
{
    Regrid r;
    std::vector<double const*> bx;
    for (auto const& data : r.ld.data)
        bx.push_back(data.Bx.v.data());

    auto level = r.h.levels[0];
    for (auto& patch : level.patches)
        patch.id += 100; // a new level, the same boxes and owners
    auto const map = r.onto(level);

    EXPECT_TRUE(map.transfers.empty());
    ASSERT_EQ(r.ld.patches, level.patches);
    for (std::size_t i = 0; i < r.ld.data.size(); ++i)
    {
        EXPECT_EQ(r.ld.data[i].Bx.v.data(), bx[i]);           // moved, not copied
        EXPECT_EQ(r.ld.data[i].patch_ghost.items.size(), 1u); // ghosts included
    }
    EXPECT_EQ(check(r.h, r.ld), 0u);
    EXPECT_EQ(toy::Tracker::peak, static_cast<long>(level.patches.size()));
}

TEST(Regrid, someBoxesKept)
{
    Regrid r;
    // the first old patch kept, the others chopped in 1 x 2
    auto const& old = r.h.levels[0].patches;
    std::vector<Box2> boxes{old[0].box};
    for (std::size_t i = 1; i < old.size(); ++i)
        for (auto const& box : chop(old[i].box, {1, 2}))
            boxes.push_back(box);
    auto const level = make_level(0, boxes);
    auto const map   = r.onto(level);

    EXPECT_EQ(map.kept[0], 0u);
    EXPECT_EQ(r.ld.data[0].patch_ghost.items.size(), 1u);
    for (auto const& t : map.transfers)
        EXPECT_NE(t.from.id, old[0].id);
    EXPECT_EQ(check(r.h, r.ld), 0u);
    EXPECT_EQ(domain_items(r.ld), 48u);
}

TEST(Regrid, otherRanksPatchesThrow)
{
    Regrid r;
    auto const boxes = chop(r.h.domain, {2, 2});
    std::vector<int> ranks(boxes.size(), 0);
    ranks.back()   = 1;
    auto const map = make_regrid_map(r.h.levels[0], make_level(0, boxes, ranks), 0);
    EXPECT_THROW((void)regrid(map, std::move(r.ld), make_patch_data), std::runtime_error);
}

TEST(Regrid, sameDecomposition)
{
    Regrid r;
    auto level = r.h.levels[0];
    EXPECT_TRUE(same_decomposition(level, r.h.levels[0]));
    level.patches.back().rank = 1;
    EXPECT_FALSE(same_decomposition(level, r.h.levels[0]));
    EXPECT_FALSE(same_decomposition(make_level(0, chop(r.h.domain, {2, 2})), r.h.levels[0]));
}

} // namespace damn::amr


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
