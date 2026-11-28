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

// each allocation spec: growing past the capacity keeps the values (a regrow: a new buffer, a
//  device copy for device memory), copies between specs and copy construction keep them too
template<typename V>
void check_vector_spec()
{
    V a(4, 1.);
    for (std::size_t i = 0; i < 4; ++i)
        a[i] = static_cast<double>(i);
    a.resize(100);
    EXPECT_EQ(a.size(), 100u);
    a.reserve(1000);
    EXPECT_GE(a.capacity(), 1000u);
    EXPECT_EQ(a.size(), 100u);

    damn::Vector<double> host;
    host.copy_from(a);
    V const b{a};
    V c;
    c.copy_from(host);
    V d(2);
    d = a;
    EXPECT_GE(host.capacity(), a.capacity()); // copies keep the capacity
    EXPECT_GE(b.capacity(), a.capacity());
    EXPECT_GE(d.capacity(), a.capacity());
    EXPECT_EQ(d.size(), a.size());
    for (std::size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(a[i], static_cast<double>(i));
        EXPECT_EQ(host[i], static_cast<double>(i));
        EXPECT_EQ(b[i], static_cast<double>(i));
        EXPECT_EQ(c[i], static_cast<double>(i));
        EXPECT_EQ(d[i], static_cast<double>(i));
    }

    a.fill(3.);
    EXPECT_EQ(a[99], 3.);
}

// host copies copy construct (types without a default constructor or copy assignment), as
//  std::vector's copy
struct CopyOnly
{
    explicit CopyOnly(int const i_)
        : i{i_}
    {
    }
    CopyOnly(CopyOnly const&)            = default;
    CopyOnly& operator=(CopyOnly const&) = delete;
    int i;
};

template<typename V>
void check_copies_construct()
{
    V a;
    a.emplace_back(1);
    a.emplace_back(2);
    V const b{a};
    EXPECT_EQ(b.size(), 2u);
    EXPECT_EQ(b[1].i, 2);
}

// also in device memory: element copies on the host, a device copy is for trivially copyable
//  values only
struct NotTrivial : CopyOnly
{
    using CopyOnly::CopyOnly;
    NotTrivial(NotTrivial const& that)
        : CopyOnly{that.i}
    {
    }
};

TEST(DamnCore, vector_copies_construct)
{
    check_copies_construct<damn::Vector<CopyOnly>>();
    check_copies_construct<damn::Vector<NotTrivial>>();
#if DAMN_HAVE_MKN_GPU
    check_copies_construct<damn::Vector<NotTrivial, damn::Allocators::device>>();
#endif
}

// after period uses under its percentile, capacity drops to realloc_to of itself
TEST(DamnCore, minimizing_vector_shrinks)
{
    damn::MinimizingVector<double> m;
    m.get(1000);
    auto const big = m.capacity();
    for (std::size_t i = 0; i < m.period; ++i)
        m.get(10);
    EXPECT_LT(m.capacity(), big);
    EXPECT_EQ(m.size(), 10u);
}

TEST(DamnCore, vector_specs)
{
    check_vector_spec<damn::Vector<double>>();
    check_vector_spec<damn::Vector<double, damn::Allocators::aligned>>();
    check_vector_spec<damn::Vector<double, damn::AllocationSpec{damn::Allocators::cpu, false}>>();
#if DAMN_HAVE_MKN_GPU
    check_vector_spec<damn::Vector<double, damn::Allocators::device>>();
    check_vector_spec<
        damn::Vector<double, damn::AllocationSpec{damn::Allocators::device, false}>>();
#endif
}

// regrows (past the capacity: a new buffer, a device copy for managed memory) keep the values,
//  every column at once with resize_columns
template<typename V>
void check_regrows_keep_values()
{
    V a(4), b(4);
    for (std::size_t i = 0; i < 4; ++i)
        a[i] = static_cast<double>(i), b[i] = static_cast<double>(10 + i);
    damn::resize_columns(100, /*copy=*/true, a, b);
    EXPECT_EQ(a.size(), 100u);
    EXPECT_EQ(b.size(), 100u);
    a.reserve(1000);
    EXPECT_GE(a.capacity(), 1000u);
    EXPECT_EQ(a.size(), 100u);
    b.resize(500);
    for (std::size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(a[i], static_cast<double>(i));
        EXPECT_EQ(b[i], static_cast<double>(10 + i));
    }
}

TEST(DamnCore, regrows_keep_values)
{
    check_regrows_keep_values<damn::Vector<double>>();
    check_regrows_keep_values<damn::Vector<double, damn::Allocators::aligned>>();
#if DAMN_HAVE_MKN_GPU
    check_regrows_keep_values<
        damn::Vector<double, damn::AllocationSpec{damn::Allocators::device, false}>>();
    check_regrows_keep_values<damn::Vector<double, damn::Allocators::device>>();
#endif
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
