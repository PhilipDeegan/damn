// damn builds and works on its own, without plamn

#include "damn/damn_core.hpp"
#include "tst/damn/core/geometry/test_box_fixtures.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <type_traits>

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

// DAMN_DEBUG_DO: on unless NDEBUG; DAMN_SUPER_DEBUG_DO: only with -DDAMN_SUPER_DEBUG
TEST(DamnCore, debug_do_switches)
{
    int light = 0, heavy = 0;
    DAMN_DEBUG_DO(++light);
    DAMN_SUPER_DEBUG_DO(++heavy);
    EXPECT_EQ(light, DAMN_DEBUG);
    EXPECT_EQ(heavy, DAMN_SUPER_DEBUG);
}

// DAMN_DAO_FN: the DAO's template arguments deduced from the arguments, trailing members
//  defaulted, and the DAO itself still accepted
namespace damn::test
{
template<std::size_t dim>
struct Scaled
{
    std::array<int, dim> const& values;
    int factor = 2;
};

template<std::size_t dim>
std::array<int, dim> scaled(Scaled<dim> const& in)
{
    auto out = in.values;
    for (auto& v : out)
        v *= in.factor;
    return out;
}

DAMN_DAO_RET_FN(inline, scaled, Scaled)

// no return value: the overload returns void too
template<std::size_t dim>
void scale(Scaled<dim> const& in, std::array<int, dim>& out)
{
    out = scaled(in);
}
template<std::size_t dim>
struct ScaleInto
{
    Scaled<dim> in;
    std::array<int, dim>& out;
};
template<std::size_t dim>
void scale_into(ScaleInto<dim> const& s)
{
    scale(s.in, s.out);
}
DAMN_DAO_FN(inline, scale_into, ScaleInto)
} // namespace damn::test

TEST(DamnCore, dao_fn_builds_the_dao_from_arguments)
{
    std::array const v{1, 2, 3};
    EXPECT_EQ(damn::test::scaled(v), (std::array{2, 4, 6}));    // factor defaulted, dim 3
    EXPECT_EQ(damn::test::scaled(v, 3), (std::array{3, 6, 9})); // factor given
    EXPECT_EQ(damn::test::scaled(damn::test::Scaled{v, 4}), (std::array{4, 8, 12})); // the DAO
    std::array const w{5, 6};
    EXPECT_EQ(damn::test::scaled(w), (std::array{10, 12})); // dim 2 from the same overload

    std::array<int, 3> out{};
    static_assert(std::is_void_v<decltype(damn::test::scale_into(damn::test::Scaled{v}, out))>);
    damn::test::scale_into(damn::test::Scaled{v, 5}, out);
    EXPECT_EQ(out, (std::array{5, 10, 15}));
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
