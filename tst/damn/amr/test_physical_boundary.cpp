#include "damn/amr/hierarchy.hpp"
#include "damn/amr/physical_boundary.hpp"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

using namespace damn;
using namespace damn::amr;

namespace
{
core::Box<int, 2> box2(std::array<int, 2> const lo, std::array<int, 2> const up)
{
    return {core::Point<int, 2>{lo}, core::Point<int, 2>{up}};
}

// 20 x 10 cells, ratio 2
Hierarchy<2> domain(bool const periodic_y)
{
    Hierarchy<2> h;
    h.domain     = box2({0, 0}, {19, 9});
    h.dx         = {1., 1.};
    h.periodic   = {false, periodic_y};
    h.max_levels = 2;
    h.ratio      = 2;
    return h;
}

auto find(std::vector<PhysicalBoundaryRegion<2>> const& regions, std::int8_t const x,
          std::int8_t const y)
{
    return std::ranges::find_if(regions,
                                [&](auto const& r) { return r.sides[0] == x and r.sides[1] == y; });
}
} // namespace

TEST(PhysicalBoundary, aCornerPatchHasTwoFacesAndACorner)
{
    auto const regions = physical_boundary_regions(domain(false), 0, box2({0, 0}, {4, 4}), 2);
    ASSERT_EQ(regions.size(), 3u);
    auto const x = find(regions, -1, 0);
    ASSERT_NE(x, regions.end());
    EXPECT_EQ(x->box, box2({-2, 0}, {-1, 6}));
    EXPECT_EQ(x->codimension(), 1u);
    auto const y = find(regions, 0, -1);
    ASSERT_NE(y, regions.end());
    EXPECT_EQ(y->box, box2({0, -2}, {6, -1}));
    auto const corner = find(regions, -1, -1);
    ASSERT_NE(corner, regions.end());
    EXPECT_EQ(corner->box, box2({-2, -2}, {-1, -1}));
    EXPECT_EQ(corner->codimension(), 2u);
}

TEST(PhysicalBoundary, periodicDirectionsHaveNoneButKeepTheirGhosts)
{
    // y periodic: only the x face, over all of the patch's ghosts along y
    auto const regions = physical_boundary_regions(domain(true), 0, box2({0, 0}, {4, 4}), 2);
    ASSERT_EQ(regions.size(), 1u);
    EXPECT_EQ(regions[0].sides, (std::array<std::int8_t, 2>{-1, 0}));
    EXPECT_EQ(regions[0].box, box2({-2, -2}, {-1, 6}));
}

TEST(PhysicalBoundary, interiorAndRefinedPatches)
{
    auto const h = domain(false);
    EXPECT_TRUE(physical_boundary_regions(h, 0, box2({5, 3}, {9, 5}), 2).empty());
    // level 1 (40 x 20): a patch at the upper x face
    auto const regions = physical_boundary_regions(h, 1, box2({30, 4}, {39, 9}), 2);
    ASSERT_EQ(regions.size(), 1u);
    EXPECT_EQ(regions[0].sides, (std::array<std::int8_t, 2>{1, 0}));
    EXPECT_EQ(regions[0].box, box2({40, 2}, {41, 11}));
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
