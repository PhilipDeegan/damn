// damn builds and works on its own, without plamn

#include "damn/damn_core.hpp"
#include "tst/damn/core/geometry/test_box_fixtures.hpp"

#include "gtest/gtest.h"

namespace damn::core
{

TEST(DamnCore, neighbour_boxes_touch_middle)
{
    auto const [middle, neighbours] = get_some_boxes();
    EXPECT_EQ(neighbours.size(), 26u);
    for (auto const& nbr : neighbours)
    {
        EXPECT_FALSE(middle * nbr);
        EXPECT_TRUE(grow(middle, 1) * nbr);
    }
}

TEST(DamnCore, ndarray_vector_shape_and_zero)
{
    NdArrayVector<3, double> arr{std::array<std::uint32_t, 3>{2, 3, 4}};
    EXPECT_EQ(arr.size(), 24u);
    for (auto& v : arr)
        v = 1;
    arr.zero();
    for (auto const& v : arr)
        EXPECT_EQ(v, 0);
}

} // namespace damn::core

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
