#ifndef DAMN_CORE_SYS_STRINGS_HPP
#define DAMN_CORE_SYS_STRINGS_HPP

#include "damn/core/def.hpp"

#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <array>
#include <cstdint>
#include <string_view>
#include <cstddef>
#include <type_traits>

namespace damn::core
{
inline auto to_string(auto const& v)
{
    std::stringstream ss;
    ss << v;
    return ss.str();
}

template<typename T>
NO_DISCARD std::string to_string_with_precision(T const& a_value, std::size_t const len)
{
    std::ostringstream out;
    out.precision(len);
    out << std::fixed << a_value;
    auto str = out.str();
    // last digit may be non 0 because of rounding
    // and we know at that decimal it should be so we force it
    str.replace(str.end() - 1, str.end(), 1, '0');
    return out.str();
}

template<typename T>
NO_DISCARD auto to_string_fixed_width(T const& value, std::size_t const& precision,
                                      std::size_t const& width, char const& fill = '0')
{
    std::ostringstream out;
    out.width(width);
    out.precision(precision);
    out << std::setfill(fill) << std::fixed << value;
    return out.str();
}


template<typename T>
NO_DISCARD T from_string(std::string const& s)
{
    T t;
    std::stringstream ss(s);
    ss >> t;
    if (ss.fail())
        throw std::runtime_error("Cannot Parse T from: " + s);
    return t;
}



template<std::string_view const&... Strs>
struct join_string_views
{
    static constexpr auto impl() noexcept
    {
        constexpr std::size_t len = (Strs.size() + ... + 0);
        std::array<char, len + 1> arr{};
        auto append = [i = 0, &arr](auto const& s) mutable {
            for (auto c : s)
                arr[i++] = c;
        };
        (append(Strs), ...);
        arr[len] = 0;
        return arr;
    }
    static constexpr auto arr = impl();
    static constexpr std::string_view value{arr.data(), arr.size() - 1};
};

template<std::string_view const&... Strs>
static constexpr auto join_string_views_v = join_string_views<Strs...>::value;


template<typename T, T t>
struct to_string_view
{
    static constexpr auto size()
    {
        if constexpr (std::is_signed_v<T>)
        {
            std::size_t len = t > 0 ? 1 : 2;
            if (t < -9 or t > 9)
                for (auto n = t; n; len++, n /= 10)
                {
                }
            return len;
        }
        else
        {
            std::size_t len = 1;
            if (t > 9)
                for (auto n = t; n; len++, n /= 10)
                {
                }
            return len;
        }
    }

    static constexpr auto impl() noexcept
    {
        constexpr std::size_t len = size();
        static_assert(len == 1);

        std::array<char, len + 1> arr{};
        if (t != 0)
        {
            std::uint16_t i = len - 1;
            for (auto n = t; n; n /= 10)
                arr[i--] = "0123456789"[(t < 0 ? -1 : 1) * (n % 10)];
            if (t < 0)
                arr[i--] = '-';
        }
        else
        {
            arr[0] = '0';
        }
        arr[len] = 0;
        return arr;
    }

    static constexpr auto arr = impl();
    static constexpr std::string_view value{arr.data(), arr.size() - 1};
};

template<typename T, T t>
static constexpr auto to_string_view_v = to_string_view<T, t>::value;

} // namespace damn::core

#endif // DAMN_CORE_SYS_STRINGS_HPP
