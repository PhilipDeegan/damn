#ifndef DAMN_CORE_MATH_NUMERIC_HPP
#define DAMN_CORE_MATH_NUMERIC_HPP

#include "damn/core/def.hpp"
#include "damn/core/meta/for_N.hpp"

#include <array>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <type_traits>

namespace damn::core
{
template<typename Container, typename Multiplies = typename Container::value_type>
NO_DISCARD Multiplies product(Container const& container, Multiplies mul = 1) DAMN_ALL
{
    // std accumulate doesn't exist on GPU
    for (auto const& v : container)
        mul *= v;
    return mul;
}

template<typename Container, typename Return = typename Container::value_type>
NO_DISCARD Return sum(Container const& container, Return r = 0) DAMN_HST
{
    return std::accumulate(container.begin(), container.end(), r);
}

template<typename T>
NO_DISCARD T norm(std::array<T, 3> vec)
{
    auto squarreSum = std::inner_product(std::begin(vec), std::end(vec), std::begin(vec), 0.);
    return std::sqrt(squarreSum);
}


template<typename Container, typename F>
NO_DISCARD auto sum_from(Container&& container, F fn)
{
    using value_type  = std::decay_t<Container>::value_type;
    using return_type = std::decay_t<std::invoke_result_t<F const&, value_type const&>>;
    return_type sum   = 0;
    for (auto const& el : container)
        sum += fn(el);
    return sum;
}


template<typename T>
auto constexpr pow(T const v, std::uint16_t const power) DAMN_ALL
{
    T out = 1;
    for (std::uint16_t i = 0; i < power; ++i)
        out *= v;
    return out;
}


auto constexpr static accessor = [](auto const& v, auto const& i) -> auto& { return v[i]; };

template<typename Container, typename Fn = decltype(accessor)>
NO_DISCARD auto constexpr max_from(Container const& container, Fn fn = accessor)
{
    assert(container.size());
    auto t = fn(container, 0);
    for (std::size_t i = 1; i < container.size(); ++i)
        if (auto const& e = fn(container, i); e > t)
            t = e;
    return t;
}

template<typename Container, typename Fn = decltype(accessor)>
NO_DISCARD auto constexpr min_from(Container const& container, Fn fn = accessor)
{
    assert(container.size());
    auto t = fn(container, 0);
    for (std::size_t i = 1; i < container.size(); ++i)
        if (auto const& e = fn(container, i); e < t)
            t = e;
    return t;
}


template<typename SignedInt, typename UnsignedInt>
bool diff_sign_int_equals(SignedInt const& i0, UnsignedInt const& i1) DAMN_ALL
{
    static_assert(std::is_unsigned_v<UnsignedInt>);
    static_assert(std::is_signed_v<SignedInt>);
    static_assert(sizeof(UnsignedInt) >= sizeof(SignedInt), "Bad int comparison!");
    if (i0 < 0)
        return false;
    return static_cast<UnsignedInt>(i0) == i1;
}


template<typename Int0, typename Int1>
bool int_equals(Int0 const& i0, Int1 const& i1) DAMN_ALL
{
    if constexpr (std::is_same_v<Int0, Int1>)
        return i0 == i1;
    else
    {
        if constexpr (std::is_unsigned_v<Int0> and std::is_signed_v<Int1>)
            return diff_sign_int_equals(i1, i0);
        if constexpr (std::is_unsigned_v<Int1> and std::is_signed_v<Int0>)
            return diff_sign_int_equals(i0, i1);
    }
    // reaching here == compiler error
}


auto inline float_equals(float const& a, float const& b, float diff = 1e-6)
{
    return std::abs(a - b) < diff;
}

auto inline float_equals(double const& a, double const& b, double diff = 1e-12)
{
    return std::abs(a - b) < diff;
}

template<std::size_t S>
bool inline float_equals(std::array<double, S> const& a, std::array<double, S> const& b,
                         double diff = 1e-15)
{
    return for_N_all<S>([&](auto i) { return float_equals(a[i], b[i], diff); });
}

auto inline float_not_equals(double const& a, double const& b, double diff = 1e-12)
{
    return std::abs(a - b) > diff;
}

template<typename F0, typename F1>
auto any_float_eq(F0 const f0, F1 const f1, double diff = 1e-12)
{
    if constexpr (sizeof(F0) == sizeof(F1))
    {
        if (sizeof(F0) == 4)
            diff *= 1e6;
        return float_equals(f0, f1, diff);
    }
    else
    {
        return float_equals(static_cast<double>(f0), static_cast<double>(f1), diff * 1e6);
    }
}

template<typename V0, typename V1>
struct FloatComparator
{
    bool operator()()
    {
        for (std::size_t i = start; i < end; ++i)
        {
            auto ret  = std::abs(v0[i] - v1[i]);
            auto pred = ret > diff;
            eq &= pred;
            max_diff = ret > max_diff ? ret : max_diff;
        }
        return eq;
    }


    V0 const& v0;
    V1 const& v1;
    double const diff = 1e-15;
    std::size_t start = 0, end;

    bool eq         = 1;
    double max_diff = 0;
};


template<typename D>
struct SetEqual
{
    using value_type = D;

    void operator()(auto const& d0) { d = d0; }

    D& d;
};

template<typename D>
struct PlusEquals
{
    using value_type = D;

    void operator()(auto const& d0) { d += d0; }
    D& d;
};

template<typename D>
struct SetMax
{
    using value_type = D;

    void operator()(auto& d0) { d = std::max(d, d0); }
    D& d;
};

template<typename D>
struct PlusEqualsProduct
{
    void operator()(auto& d0, auto& o) { d += d0 * o; }
    D& d;
};

} // namespace damn::core

#endif // DAMN_CORE_MATH_NUMERIC_HPP
