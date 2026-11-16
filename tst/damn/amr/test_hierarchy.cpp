// damn::amr::Hierarchy: level index spaces, decomposition, level replacement, periodicity

#include "damn/amr/hierarchy.hpp"
#include "damn/core/geometry/box_spans.hpp"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <map>
#include <vector>

namespace damn::amr
{
namespace
{
    template<std::size_t dim>
    auto box(std::array<int, dim> const& lower, std::array<int, dim> const& upper)
    {
        return core::Box<int, dim>{lower, upper};
    }

    // 10 x 8 cells from (0, 0), dx (.1, .2), origin (1, 2)
    Hierarchy<2> hierarchy_2d(std::array<bool, 2> const periodic = {})
    {
        Hierarchy<2> h;
        h.domain     = box<2>({0, 0}, {9, 7});
        h.dx         = {.1, .2};
        h.origin     = core::Point<double, 2>{std::array{1., 2.}};
        h.periodic   = periodic;
        h.max_levels = 3;
        return h;
    }
} // namespace


// ---------------------------------------------------------------- box refine / coarsen

static_assert(core::floor_div(3, 2) == 1);
static_assert(core::floor_div(-1, 2) == -1);
static_assert(core::floor_div(-2, 2) == -1);
static_assert(core::floor_div(-3, 2) == -2);
static_assert(core::floor_div(0, 2) == 0);

TEST(BoxRatio, floorDivPerElement)
{
    EXPECT_EQ(core::floor_div(core::Point<int, 3>{std::array{-3, 0, 3}}, 2),
              (core::Point<int, 3>{std::array{-2, 0, 1}}));
}

TEST(BoxRatio, refineCoversTheSameRegion)
{
    EXPECT_EQ(core::refine(box<1>({-1}, {1}), 2), box<1>({-2}, {3}));
    EXPECT_EQ(core::refine(box<2>({0, 3}, {4, 3}), 4), box<2>({0, 12}, {19, 15}));
}

TEST(BoxRatio, coarsenRoundsOutwardsAlsoForNegativeCells)
{
    EXPECT_EQ(core::coarsen(box<2>({-3, -1}, {2, 5}), 2), box<2>({-2, -1}, {1, 2}));
    EXPECT_EQ(core::coarsen(box<1>({1}, {2}), 2), box<1>({0}, {1})); // straddles two coarse cells
}

TEST(BoxRatio, coarsenOfRefineIsIdentity)
{
    auto const b = box<3>({-5, 0, 7}, {-1, 3, 7});
    for (int const r : {2, 3, 4})
        EXPECT_EQ(core::coarsen(core::refine(b, r), r), b);
}


// ---------------------------------------------------------------- levels

TEST(Hierarchy, levelIndexSpacesAndMeshSizes)
{
    auto const h = hierarchy_2d();

    EXPECT_EQ(ratio_to_level_0(h, 0), 1);
    EXPECT_EQ(ratio_to_level_0(h, 2), 4);
    EXPECT_EQ(level_domain(h, 1), box<2>({0, 0}, {19, 15}));
    EXPECT_EQ(level_domain(h, 2), box<2>({0, 0}, {39, 31}));

    auto const dx = mesh_size(h, 2);
    EXPECT_DOUBLE_EQ(dx[0], .025);
    EXPECT_DOUBLE_EQ(dx[1], .05);
}

TEST(Hierarchy, positionOfCellLowerCorners)
{
    auto const h = hierarchy_2d();

    auto const x0 = position(h, 0, core::Point<int, 2>{std::array{3, 1}});
    EXPECT_DOUBLE_EQ(x0[0], 1.3);
    EXPECT_DOUBLE_EQ(x0[1], 2.2);

    // the same corner on level 1
    auto const x1 = position(h, 1, core::Point<int, 2>{std::array{6, 2}});
    EXPECT_DOUBLE_EQ(x1[0], x0[0]);
    EXPECT_DOUBLE_EQ(x1[1], x0[1]);
}


// ---------------------------------------------------------------- decomposition

TEST(Hierarchy, chopCoversTheBoxWithDisjointBoxes)
{
    auto const domain = box<2>({0, 0}, {9, 7});
    auto const boxes  = chop(domain, {4, 4}); // 3 x 2, last column 2 cells wide

    ASSERT_EQ(boxes.size(), 6u);
    EXPECT_EQ(boxes[0], box<2>({0, 0}, {3, 3}));
    EXPECT_EQ(boxes[2], box<2>({8, 0}, {9, 3})); // x fastest
    EXPECT_EQ(boxes[5], box<2>({8, 4}, {9, 7}));

    std::size_t cells = 0;
    for (std::size_t i = 0; i < boxes.size(); ++i)
    {
        cells += boxes[i].size();
        for (std::size_t j = i + 1; j < boxes.size(); ++j)
            EXPECT_FALSE(boxes[i] * boxes[j]);
    }
    EXPECT_EQ(cells, domain.size());
}

TEST(Hierarchy, uniformLevel0PartitionsCellsOverRanks)
{
    auto const level = uniform_level_0(hierarchy_2d(), {4, 4}, 2);

    // 3 x 2 patches of 16, 16 and 8 cells per row: 80 cells, 40 a rank (partition.hpp)
    ASSERT_EQ(level.patches.size(), 6u);
    std::array<std::size_t, 2> cells{};
    for (auto const& p : level.patches)
    {
        ASSERT_TRUE(p.rank == 0 or p.rank == 1);
        cells[p.rank] += p.box.size();
    }
    EXPECT_LE(std::max(cells[0], cells[1]) - 40, 16u);

    auto const local        = local_patches(level, 1);
    std::size_t local_cells = 0;
    for (auto const& p : local)
    {
        EXPECT_EQ(p.rank, 1);
        local_cells += p.box.size();
    }
    EXPECT_EQ(local_cells, cells[1]);
    EXPECT_EQ(patch_with_id(level, 4).box, level.patches[4].box);
}


// ---------------------------------------------------------------- level replacement

TEST(Hierarchy, setLevelReplacesAndDropsFinerLevels)
{
    auto h = hierarchy_2d();
    set_level(h, uniform_level_0(h, {5, 4}));
    set_level(h, make_level<2>(1, {box<2>({4, 4}, {11, 9})}));
    set_level(h, make_level<2>(2, {box<2>({10, 10}, {17, 15})}));
    EXPECT_EQ(n_levels(h), 3);
    EXPECT_EQ(finest_level(h), 2);

    set_level(h, make_level<2>(1, {box<2>({0, 0}, {7, 7})})); // regrid level 1
    EXPECT_EQ(n_levels(h), 2);
    EXPECT_EQ(h.levels[1].patches[0].box, box<2>({0, 0}, {7, 7}));

    remove_finer_levels(h, 0);
    EXPECT_EQ(n_levels(h), 1);
    EXPECT_TRUE(level_exists(h, 0));
    EXPECT_FALSE(level_exists(h, 1));
}

TEST(Hierarchy, setLevelRejectsInvalidLevels)
{
    auto h = hierarchy_2d();

    // no level 0 yet
    EXPECT_THROW(set_level(h, make_level<2>(1, {box<2>({0, 0}, {1, 1})})), std::runtime_error);
    // outside the domain
    EXPECT_THROW(set_level(h, make_level<2>(0, {box<2>({8, 0}, {10, 7})})), std::runtime_error);
    // overlapping patches
    EXPECT_THROW(set_level(h, make_level<2>(0, {box<2>({0, 0}, {5, 7}), box<2>({5, 0}, {9, 7})})),
                 std::runtime_error);
    // beyond max_levels
    set_level(h, uniform_level_0(h, {10, 8}));
    set_level(h, make_level<2>(1, {box<2>({0, 0}, {3, 3})}));
    set_level(h, make_level<2>(2, {box<2>({0, 0}, {3, 3})}));
    EXPECT_THROW(set_level(h, make_level<2>(3, {box<2>({0, 0}, {3, 3})})), std::runtime_error);
}


// ---------------------------------------------------------------- periodicity

TEST(Hierarchy, periodicShiftsAreTheNeighbouringDomainImages)
{
    EXPECT_TRUE(periodic_shifts(hierarchy_2d(), 0).empty());

    auto const x_only = periodic_shifts(hierarchy_2d({true, false}), 1);
    ASSERT_EQ(x_only.size(), 2u);
    EXPECT_EQ(x_only[0], (core::Point<int, 2>{std::array{-20, 0}}));
    EXPECT_EQ(x_only[1], (core::Point<int, 2>{std::array{20, 0}}));

    auto const both = periodic_shifts(hierarchy_2d({true, true}), 0);
    EXPECT_EQ(both.size(), 8u);
    EXPECT_TRUE(std::find(both.begin(), both.end(), core::Point<int, 2>{std::array{-10, 8}})
                != both.end());
}

TEST(Hierarchy, coveredByLevelIncludesPeriodicImages)
{
    auto h = hierarchy_2d({true, false});
    set_level(h, uniform_level_0(h, {5, 8})); // two patches side by side

    auto const& left = h.levels[0].patches[0].box; // (0,0) (4,7)
    auto const ghost = core::grow(left, 1);        // (-1,-1) (5,8)

    std::size_t cells = 0;
    for (auto const& b : covered_by_level(h, 0, ghost))
        cells += b.size();
    // all but the y ghost rows: x is periodic, y is not
    EXPECT_EQ(cells, 7u * 8u);
}

// every cell some patch image holds in its grown box has exactly one owner: the patches' grown
//  boxes less held_by_earlier partition them, periodic images included (a cell counted in the
//  domain's index space)
TEST(Hierarchy, heldByEarlierLeavesEveryHeldCellOneOwner)
{
    for (auto const periodic : {false, true})
        for (int const width : {1, 2, 3})
        {
            auto h = hierarchy_2d({periodic, periodic});
            set_level(h, uniform_level_0(h, {5, 4}));
            // level 1 (20 x 16): patches touching, one cell apart, and further than 2 * width
            set_level(h, make_level<2>(1, {box<2>({0, 0}, {5, 6}), box<2>({6, 0}, {9, 4}),
                                           box<2>({11, 0}, {14, 4}), box<2>({6, 9}, {19, 15})}));
            for (int const ln : {0, 1})
            {
                auto const shape     = level_domain(h, ln).shape();
                auto const canonical = [&](core::Point<int, 2> p) {
                    if (periodic)
                        for (std::size_t i = 0; i < 2; ++i)
                            p[i] = ((p[i] % shape[i]) + shape[i]) % shape[i];
                    return std::array{p[0], p[1]};
                };
                std::map<std::array<int, 2>, int> owners;
                for (auto const& patch : h.levels[ln].patches)
                {
                    auto const held = held_by_earlier(h, ln, patch.box, width);
                    core::for_each_point(core::grow(patch.box, width), [&](auto const& c) {
                        auto& n = owners[canonical(c)];
                        n += !isIn(c, held);
                    });
                }
                for (auto const& [cell, n] : owners)
                    EXPECT_EQ(n, 1) << "level " << ln << " width " << width << " periodic "
                                    << periodic << " cell " << cell[0] << "," << cell[1];
            }
        }
}

} // namespace damn::amr


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
