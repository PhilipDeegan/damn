// damn::amr::regrid on one rank: the toy level (8 x 6 cells, 3 x 4 patches) onto other
// decompositions of the same cells

#include "damn/amr/hierarchy.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/regrid.hpp"
#include "tst/damn/amr/toy.hpp"
#include "tst/damn/amr/toy_regrid.hpp"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string_view>
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
    for (auto const& t : map.transfers) // the kept patch only gives its border points
        if (t.from.id == old[0].id)
            EXPECT_FALSE(t.cells.has_value());
    EXPECT_EQ(check(r.h, r.ld), 0u);
    EXPECT_EQ(domain_items(r.ld), 48u);
}

namespace
{
    // old: the left half of the 8 x 6 domain (periodic in x); new: that half, kept or chopped,
    //  and the right half, new cells
    struct Grows
    {
        Hierarchy<2> h = [] {
            auto h = make_hierarchy({true, false});
            set_level(h, make_level(0, std::vector{Box2{std::array{0, 0}, std::array{3, 5}}}));
            return h;
        }();
        LevelData_t ld = allocate(h.levels[0], 0, make_patch_data);
        Grows() { fill(h, ld); }
    };

    // the new right patch's domain points from the old patch: its x = 4 border (shared, no
    //  shared cell) and its x = 8 border (the image of x = 0), the rest untouched (NaN)
    void expect_borders_from_the_old_patch(Hierarchy<2> const& h, Patch<2> const& right,
                                           toy::PatchData& data)
    {
        for_each_field(data, [&](std::string_view const name, toy::Field& f) {
            core::for_each_point(domain_box(f, right.box), [&](auto const& p) {
                bool const border = f.primal[0] and (p[0] == 4 or p[0] == 8);
                if (border)
                    EXPECT_EQ(value(f, right.box, p), g_of(h, name, p)) << name << " " << p;
                else
                    EXPECT_TRUE(std::isnan(value(f, right.box, p))) << name << " " << p;
            });
        });
        EXPECT_TRUE(data.domain.items.empty()); // no shared cell
    }
} // namespace

TEST(Regrid, newCellsTakeTheBorderPointsTheyShareKeptOrNot)
{
    Box2 const right{std::array{4, 0}, std::array{7, 5}};
    for (bool const kept : {true, false})
    {
        Grows g;
        std::vector<Box2> boxes{right};
        if (kept)
            boxes.push_back(g.h.levels[0].patches[0].box);
        else
            for (auto const& b : chop(g.h.levels[0].patches[0].box, {4, 3}))
                boxes.push_back(b);
        auto const level = make_level(0, boxes);
        auto const map
            = make_regrid_map(g.h.levels[0], level, 0, periodic_shifts(g.h, 0));
        EXPECT_EQ(map.kept[1] != not_local, kept);
        g.ld = regrid(map, std::move(g.ld), make_patch_data);

        expect_borders_from_the_old_patch(g.h, level.patches[0], g.ld.data[0]);
        // the left half: its old data
        for (std::size_t i = 1; i < level.patches.size(); ++i)
            for_each_field(g.ld.data[i], [&](std::string_view const name, toy::Field& f) {
                auto const& cells = level.patches[i].box;
                core::for_each_point(domain_box(f, cells), [&](auto const& p) {
                    EXPECT_EQ(value(f, cells, p), g_of(g.h, name, p)) << name << " " << p;
                });
            });
    }
}

TEST(Regrid, moveOnlyPatchData)
{
    Grows g;
    LevelData<2, std::unique_ptr<toy::PatchData>> ld{0, g.ld.patches, {}};
    for (auto& d : g.ld.data)
        ld.data.push_back(std::make_unique<toy::PatchData>(std::move(d)));

    Box2 const right{std::array{4, 0}, std::array{7, 5}};
    auto const level = make_level(0, std::vector{right, g.h.levels[0].patches[0].box});
    auto const map   = make_regrid_map(g.h.levels[0], level, 0, periodic_shifts(g.h, 0));
    ld               = regrid(map, std::move(ld), [](Patch<2> const& p) {
        return std::make_unique<toy::PatchData>(make_patch_data(p));
    });
    expect_borders_from_the_old_patch(g.h, level.patches[0], *ld.data[0]);
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
