#include "damn/amr/clustering.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

namespace damn::amr
{
namespace
{
    using Box2   = core::Box<int, 2>;
    using Point2 = core::Point<int, 2>;

    Box2 box(int x0, int y0, int x1, int y1)
    {
        return Box2{std::array{x0, y0}, std::array{x1, y1}};
    }
    Point2 cell(int x, int y)
    {
        return Point2{std::array{x, y}};
    }

    // n x n cells, one patch
    Hierarchy<2> level0(int const n, bool const periodic)
    {
        Hierarchy<2> h;
        h.domain     = box(0, 0, n - 1, n - 1);
        h.dx         = {.1, .1};
        h.periodic   = {periodic, periodic};
        h.max_levels = 3;
        set_level(h, make_level(0, std::vector{h.domain}));
        return h;
    }

    std::vector<Box2> clusters(Hierarchy<2> const& h, int const ln,
                               std::vector<Point2> const& tagged,
                               ClusteringOptions<2> const& options = {})
    {
        auto boxes = cluster(h, ln, tagged_tiles(h, ln, tagged, options), options);
        std::sort(boxes.begin(), boxes.end(), [](auto const& a, auto const& b) {
            return a.lower.toArray() < b.lower.toArray();
        });
        return boxes;
    }

    std::size_t cells(std::vector<Box2> const& boxes)
    {
        std::size_t n = 0;
        for (auto const& b : boxes)
            n += b.size();
        return n;
    }
} // namespace


TEST(Clustering, aTaggedCellIsItsTileRefined)
{
    auto const h = level0(32, true);
    EXPECT_EQ(clusters(h, 0, {cell(12, 12)}), (std::vector{box(16, 16, 31, 31)}));
}

TEST(Clustering, theTagBufferReachesTheNextTilesWhichCoalesce)
{
    auto const h = level0(32, true);
    // 7 +- 1 spans tiles 0 and 1 in both directions: 4 tiles, one box
    EXPECT_EQ(clusters(h, 0, {cell(7, 7)}), (std::vector{box(0, 0, 31, 31)}));
    // no buffer: one tile
    ClusteringOptions<2> none;
    none.tag_buffer = 0;
    EXPECT_EQ(clusters(h, 0, {cell(7, 7)}, none), (std::vector{box(0, 0, 15, 15)}));
}

TEST(Clustering, tilesInARowCoalesceIntoOneBox)
{
    auto const h = level0(32, true);
    EXPECT_EQ(clusters(h, 0, {cell(4, 4), cell(12, 4), cell(20, 4)}),
              (std::vector{box(0, 0, 47, 15)}));
    // an L of 3 tiles: two boxes, every cell once
    auto const l = clusters(h, 0, {cell(4, 4), cell(12, 4), cell(4, 12)});
    EXPECT_EQ(l.size(), 2u);
    EXPECT_EQ(cells(l), 3u * 16 * 16);
}

TEST(Clustering, periodicImagesWrapIntoTheDomain)
{
    auto const h = level0(32, true);
    // 0 +- 1 reaches cells 31: the four corner tiles, not adjacent in the domain
    auto const boxes = clusters(h, 0, {cell(0, 0)});
    EXPECT_EQ(boxes, (std::vector{box(0, 0, 15, 15), box(0, 48, 15, 63), box(48, 0, 63, 15),
                                  box(48, 48, 63, 63)}));
}

TEST(Clustering, theDomainClipsTilesNotPeriodicDropsOutside)
{
    auto const h = level0(30, false); // tiles of 8: the last is 6 wide
    EXPECT_EQ(clusters(h, 0, {cell(29, 29)}), (std::vector{box(48, 48, 59, 59)}));
    EXPECT_EQ(clusters(h, 0, {cell(0, 0)}), (std::vector{box(0, 0, 15, 15)}));
}

TEST(Clustering, levelTwoBoxesAreNestedInLevelOne)
{
    auto h = level0(32, true);
    set_level(h, make_level(1, std::vector{box(16, 16, 47, 47)})); // coarse 8..23
    // a tag at level 1's corner: its tile is cut to the cells a buffer inside level 1
    auto const boxes = clusters(h, 1, {cell(16, 16)});
    ASSERT_FALSE(boxes.empty());
    auto const nest = core::refine(box(17, 17, 46, 46), h.ratio);
    for (auto const& b : boxes)
    {
        auto const overlap = b * nest;
        ASSERT_TRUE(overlap.has_value());
        EXPECT_EQ(*overlap, b) << b;
    }
    // the tile 16..23 (and its neighbours' cells within the buffer) less the outer cell
    EXPECT_EQ(cells(boxes), core::refine(box(17, 17, 23, 23), h.ratio).size());

    ClusteringOptions<2> two;
    two.nesting_buffer = 2;
    EXPECT_EQ(cells(clusters(h, 1, {cell(16, 16)}, two)),
              core::refine(box(18, 18, 23, 23), h.ratio).size());
}

TEST(Clustering, ratioThreeRefinesTilesByThreeAndNestsLevelTwo)
{
    auto h  = level0(32, true);
    h.ratio = 3;
    EXPECT_EQ(clusters(h, 0, {cell(12, 12)}), (std::vector{box(24, 24, 47, 47)}));

    set_level(h, make_level(1, std::vector{box(24, 24, 71, 71)})); // coarse 8..23
    auto const boxes = clusters(h, 1, {cell(24, 24)});
    ASSERT_FALSE(boxes.empty());
    auto const nest = core::refine(box(25, 25, 70, 70), h.ratio);
    for (auto const& b : boxes)
    {
        auto const overlap = b * nest;
        ASSERT_TRUE(overlap.has_value());
        EXPECT_EQ(*overlap, b) << b;
    }
    EXPECT_EQ(cells(boxes), core::refine(box(25, 25, 31, 31), h.ratio).size());
}

TEST(Clustering, nothingTaggedNoBoxes)
{
    auto const h = level0(32, true);
    EXPECT_TRUE(clusters(h, 0, {}).empty());
}

} // namespace damn::amr


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
