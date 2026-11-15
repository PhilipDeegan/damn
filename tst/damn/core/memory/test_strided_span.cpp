// damn::core::StridedSpan: a member of an array of structs, a component of an array of
// std::array and a column, read and written in place; a stride that is not a whole number of
// values is refused

#include "damn/core/memory/strided_span.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace damn::core
{
namespace
{
    struct P
    {
        double w;
        std::array<int, 3> c;
        std::array<double, 2> d;
    };
} // namespace

TEST(StridedSpan, aMemberOfAnArrayOfStructs)
{
    std::vector<P> ps(5);
    for (std::size_t i = 0; i < ps.size(); ++i)
        ps[i] = P{1. * i, {int(i), 0, 0}, {10. * i, 20. * i}};

    auto const w  = strided_span(&ps[0].w, ps.size(), sizeof(P));
    auto const dy = strided_span(&ps[0].d[1], ps.size(), sizeof(P));
    auto const cx = strided_span(&ps[0].c[0], ps.size(), sizeof(P));
    EXPECT_FALSE(w.contiguous());
    EXPECT_EQ(w.stride_values(), sizeof(P) / sizeof(double));
    EXPECT_EQ(cx.stride_values(), sizeof(P) / sizeof(int));
    for (std::size_t i = 0; i < ps.size(); ++i)
    {
        EXPECT_EQ(w[i], 1. * i);
        EXPECT_EQ(dy[i], 20. * i);
        EXPECT_EQ(cx[i], int(i));
    }
    dy[3] = -1;
    EXPECT_EQ(ps[3].d[1], -1.); // in place

    std::vector<P> const cps = ps;
    auto const cw            = strided_span(&cps[0].w, cps.size(), sizeof(P)); // const
    EXPECT_EQ(cw[4], 4.);
}

TEST(StridedSpan, aComponentOfAnArrayOfStdArrayAndAColumn)
{
    std::vector<std::array<double, 3>> v(4);
    for (std::size_t i = 0; i < v.size(); ++i)
        v[i] = {1. * i, 2. * i, 3. * i};
    auto const vz = strided_span(&v[0][2], v.size(), sizeof(v[0]));
    EXPECT_EQ(vz.stride_values(), 3u);
    for (std::size_t i = 0; i < v.size(); ++i)
        EXPECT_EQ(vz[i], 3. * i);

    std::vector<double> column{1, 2, 3};
    auto const c = strided_span(column.data(), column.size(), sizeof(double));
    EXPECT_TRUE(c.contiguous());
    EXPECT_EQ(c[2], 3.);
}

TEST(StridedSpan, refusesAStrideOfPartValues)
{
    std::vector<double> d(4);
    EXPECT_THROW(std::ignore = strided_span(d.data(), d.size(), 12), std::runtime_error);
    EXPECT_THROW(std::ignore = strided_span(d.data(), d.size(), 4), std::runtime_error);
}

} // namespace damn::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
