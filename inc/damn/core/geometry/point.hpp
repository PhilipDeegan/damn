#ifndef DAMN_CORE_GEOMETRY_POINT_HPP
#define DAMN_CORE_GEOMETRY_POINT_HPP

#include "damn/core/def.hpp"
#include "damn/core/def/damn_config.hpp"

#include "mkn/kul/int.hpp"
#include "mkn/kul/meta.hpp"
#include "mkn/kul/std/array.hpp"
#include "mkn/kul/string.hpp"

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace damn
{
namespace core
{
    template<typename Type, std::size_t dim>
    class Point
    {
    public:
        static constexpr std::size_t dimension = dim;
        using value_type                       = Type;


        template<typename... Indexes>
        constexpr Point(std::tuple<Indexes...> index) DAMN_ALL
            : r{std::apply([](auto const&... args) { return std::array<Type, dim>{args...}; },
                           index)}
        {
            static_assert(sizeof...(Indexes) == dimension,
                          "Error dimension does match number of arguments");
        }


        template<typename... Indexes>
        constexpr Point(Indexes... index) DAMN_ALL : r{{static_cast<Type>(index)...}}
        {
            mkn::kul::allsame(index...);
            static_assert(sizeof...(Indexes) == dimension,
                          "Error dimension does match number of arguments");
        }


        constexpr Point(std::array<Type, dim> const& coords) DAMN_ALL : r{coords} {}

        template<mkn::kul::Subscriptable Container>
        Point(Container c) DAMN_ALL
        {
            for (std::size_t i = 0; i < dim; ++i)
            {
                r[i] = c[i];
            }
        }

        constexpr Point() { mkn::kul::fill(Type{0}, r); }

        MKN_KUL_FN_SIG_RET constexpr auto& operator[](std::size_t const i) DAMN_ALL { return r[i]; }
        MKN_KUL_FN_SIG_RET constexpr auto const& operator[](std::size_t const i) const DAMN_ALL
        {
            return r[i];
        }


        template<typename T2>
        MKN_KUL_FN_SIG_RET bool operator==(Point<T2, dim> const& p) const DAMN_ALL
        {
            bool areEqual = true;
            for (std::size_t i = 0; i < dim; ++i)
            {
                static_assert(std::is_integral_v<Type>,
                              "this function is only valid for integral type of Point");

                areEqual &= mkn::kul::int_equals((*this)[i], p[i]); // handles signed differences
            }
            return areEqual;
        }

        MKN_KUL_FN_SIG_RET bool operator!=(Point const& other) const DAMN_ALL
        {
            return !(*this == other);
        }


        // template<template<typename, std::size_t> typename Arr, typename T>
        // auto operator<(Arr<T, dim> const& arr) const DAMN_ALL
        // {
        //     return for_N_all<dim>([&](auto iDim) { return r[iDim] < arr[iDim]; });
        // }
        // auto operator<(auto const& v) const DAMN_ALL
        // {
        //     return for_N_all<dim>([&](auto iDim) { return r[iDim] < v; });
        // }
        // template<template<typename, std::size_t> typename Arr, typename T>
        // auto operator<=(Arr<T, dim> const& arr) const DAMN_ALL
        // {
        //     return for_N_all<dim>([&](auto iDim) { return r[iDim] <= arr[iDim]; });
        // }


        // template<template<typename, std::size_t> typename Arr, typename T>
        // auto operator>(Arr<T, dim> const& arr) const DAMN_ALL
        // {
        //     return for_N_all<dim>([&](auto iDim) { return r[iDim] > arr[iDim]; });
        // }
        // auto operator>(auto const& v) const DAMN_ALL
        // {
        //     return for_N_all<dim>([&](auto iDim) { return r[iDim] > v; });
        // }
        // template<template<typename, std::size_t> typename Arr, typename T>
        // auto operator>=(Arr<T, dim> const& arr) const DAMN_ALL
        // {
        //     return for_N_all<dim>([&](auto iDim) { return r[iDim] >= arr[iDim]; });
        // }
        // auto operator>=(auto const& v) const DAMN_ALL
        // {
        //     return for_N_all<dim>([&](auto iDim) { return r[iDim] >= v; });
        // }




        template<typename DestType = Type>
        MKN_KUL_FN_SIG_RET auto toArray() const DAMN_ALL
        {
            std::array<DestType, dimension> destArray;
            for (auto i = 0u; i < dimension; ++i)
            {
                destArray[i] = static_cast<DestType>(r[i]);
            }
            return destArray;
        }


        MKN_KUL_FN_SIG_RET std::vector<Type> toVector() const
        {
            return std::vector<Type>(r.data(), r.data() + dimension);
        }

        MKN_KUL_FN_SIG_RET std::string str(std::uint8_t const precision = 6,
                                           bool left_pad_positive       = true) const
            requires(std::floating_point<Type>)
        {
            std::stringstream ss;
            for (auto const v : r)
            {
                if (left_pad_positive and v >= 0)
                    ss << " ";
                ss << mkn::kul::to_string_with_precision(v, precision) << ",";
            }
            auto s = ss.str(); // drop last comma
            s.pop_back();
            return s;
        }


        MKN_KUL_FN_SIG_RET std::string str() const
            requires(!std::floating_point<Type>)
        {
            std::stringstream ss;
            ss << r[0];
            for (std::size_t i = 1; i < dim; ++i)
            {
                ss << "," << r[i];
            }
            return ss.str();
        }

        MKN_KUL_FN_SIG_RET static Point fromString(std::string csv)
        {
            Point p;
            std::istringstream split(csv);
            std::vector<std::string> tokens;
            for (std::string each; std::getline(split, each, ','); tokens.push_back(each))
            {
            }
            assert(tokens.size() == dimension);
            for (std::size_t i = 0; i < tokens.size(); i++)
            {
                std::stringstream ss;
                ss << tokens[i];
                ss >> p.r[i];
            }
            return p;
        }



        template<template<typename, std::size_t> typename Arr, typename T>
        auto& operator+=(Arr<T, dim> const& value) DAMN_ALL
        {
            for (auto iDim = 0u; iDim < dim; ++iDim)
                r[iDim] += value[iDim];
            return *this;
        }

        template<template<typename, std::size_t> typename Arr, typename T>
        auto& operator-=(Arr<T, dim> const& value) DAMN_ALL
        {
            for (auto iDim = 0u; iDim < dim; ++iDim)
                r[iDim] -= value[iDim];
            return *this;
        }


        auto& operator+=(Type const& value) DAMN_ALL
        {
            for (auto iDim = 0u; iDim < dim; ++iDim)
                r[iDim] += value;
            return *this;
        }
        // auto& operator+=(std::array<Type, dim> const& value) DAMN_ALL
        // {
        //     for (auto iDim = 0u; iDim < dim; ++iDim)
        //         r[iDim] += value[iDim];
        //     return *this;
        // }

        auto& operator-=(Type const& value) DAMN_ALL
        {
            for (auto iDim = 0u; iDim < dim; ++iDim)
                r[iDim] -= value;
            return *this;
        }
        // auto& operator-=(std::array<Type, dim> const& value) DAMN_ALL
        // {
        //     for (auto iDim = 0u; iDim < dim; ++iDim)
        //         r[iDim] -= value[iDim];
        //     return *this;
        // }

        auto& operator*=(Type const& value) DAMN_ALL
        {
            for (auto iDim = 0u; iDim < dim; ++iDim)
                r[iDim] *= value;
            return *this;
        }
        auto& operator*=(std::array<Type, dim> const& value) DAMN_ALL
        {
            for (auto iDim = 0u; iDim < dim; ++iDim)
                r[iDim] *= value[iDim];
            return *this;
        }

        auto operator+(Type const& value) const DAMN_ALL { return Point{r} += value; }
        auto operator+(std::array<Type, dim> const& value) const DAMN_ALL
        {
            return Point{r} += value;
        }
        auto operator+(Point<Type, dim> const& value) const DAMN_ALL
        {
            return (*this) + value.r;
        }

        auto operator-(Type const& value) const DAMN_ALL { return Point{r} -= value; }
        auto operator-(std::array<Type, dim> const& value) const DAMN_ALL
        {
            return Point{r} -= value;
        }
        auto operator-(Point<Type, dim> const& value) const DAMN_ALL
        {
            return (*this) - value.r;
        }

        auto operator*(Type const& value) const DAMN_ALL { return Point{r} *= value; }
        auto operator*(std::array<Type, dim> const& value) const DAMN_ALL
        {
            return Point{r} *= value;
        }
        auto operator*(Point<Type, dim> const& value) const DAMN_ALL
        {
            return (*this) * value.r;
        }




        // auto operator*(Type const& value) const
        // {
        //     auto copy = *this;
        //     for (auto iDim = 0u; iDim < dim; ++iDim)
        //         copy[iDim] *= value;
        //     return copy;
        // }
        // auto operator*(std::array<Type, dim> const& value) const
        // {
        //     auto copy = *this;
        //     for (auto iDim = 0u; iDim < dim; ++iDim)
        //         copy[iDim] *= value[iDim];
        //     return copy;
        // }
        // auto operator*(Point<Type, dim> const& value) const { return (*this) * value.r; }


        // Point operator%(Type const& value) const DAMN_ALL
        // {
        //     return {for_N_make_array<dim>([&](auto i) { return (*this)[i] % value; })};
        // }


        // Point operator/(Type const& value) const DAMN_ALL
        // {
        //     return {for_N_make_array<dim>([&](auto i) { return (*this)[i] / value; })};
        // }


        MKN_KUL_FN_SIG_RET constexpr auto size() const { return dim; }
        MKN_KUL_FN_SIG_RET auto data() const { return r.data(); }
        MKN_KUL_FN_SIG_RET auto begin() DAMN_ALL { return r.begin(); }
        MKN_KUL_FN_SIG_RET auto begin() const DAMN_ALL { return r.begin(); }
        MKN_KUL_FN_SIG_RET auto end() DAMN_ALL { return r.end(); }
        MKN_KUL_FN_SIG_RET auto end() const DAMN_ALL { return r.end(); }

        MKN_KUL_FN_SIG_RET auto& operator*() const DAMN_ALL { return r; }
        MKN_KUL_FN_SIG_RET auto& operator()() const DAMN_ALL { return r; }

        operator std::array<Type, dim>() const DAMN_ALL { return r; }


        template<typename To>
        auto as() const DAMN_ALL
        {
            return Point<To, dim>{this->template toArray<To>()};
        }

        auto as_unsigned() const DAMN_ALL
        {
            for (auto iDim = 0u; iDim < dim; ++iDim)
                if (r[iDim] < 0)
                    throw std::runtime_error("Cannot make unsigned from negative values");
            if constexpr (sizeof(Type) == 4)
                return as<std::uint32_t>();
            // else no return cause not yet handled
        }

        auto as_signed() const DAMN_ALL
        {
            if constexpr (sizeof(Type) == 4)
                return as<std::int32_t>();

            // else no return cause not yet handled
        }


        MKN_KUL_FN_SIG_RET constexpr Point<Type, dim> shift(std::size_t d, int offset) const
        {
            assert(d < dim);
            Point<Type, dim> result = *this;
            result[d] += static_cast<Type>(offset);
            return result;
        }


    private:
        std::array<Type, dim> r{};
    };

    template<typename... Indexes, // block constructor from use if not int/float/etc
             typename
             = typename std::enable_if<(true && ... && std::is_arithmetic_v<Indexes>), void>::type>
    Point(Indexes... indexes)
        -> Point<typename std::tuple_element<0, std::tuple<Indexes...>>::type, sizeof...(indexes)>;


    template<typename Type, std::size_t dim>
    auto& operator<<(std::ostream& os, Point<Type, dim> const& p)
    {
        os << "( ";
        for (auto& v : p)
            os << v << " ";
        os << ")";
        return os;
    }


} // namespace core
} // namespace damn

namespace damn::core
{

template<typename T0, typename... Args>
auto to_point(Args&&... args) DAMN_ALL
{
    std::array<T0, sizeof...(Args)> arr;
    std::size_t idx = -1;
    auto const set  = [&](auto& arg) { arr[++idx] = arg; };
    (set(args), ...);
    return Point{arr};
}


template<template<typename, std::size_t> typename Arr, typename T, std::size_t dim>
auto as_point(Arr<T, dim> const& arr)
{
    return Point<T, dim>{arr};
}

template<typename Type, std::size_t dim, typename Shifter>
MKN_KUL_FN_SIG_RET Point<Type, dim> shift(Point<Type, dim> const& point, Shifter const& offset)
{
    auto copy{point};
    copy += offset;
    return copy;
}


} // namespace damn::core

namespace std
{
template<typename Type, std::size_t dim>
MKN_KUL_FN_SIG_RET damn::core::Point<Type, dim> abs(damn::core::Point<Type, dim> const& point)
{
    std::array<Type, dim> postive;
    for (std::size_t i = 0; i < dim; ++i)
        postive[i] = std::abs(point[i]);
    return postive;
}


template<typename Type, std::size_t dim>
auto to_string(damn::core::Point<Type, dim> const& point)
{
    return point.str();
}


} // namespace std


#endif
