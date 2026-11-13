#ifndef DAMN_CORE_MATH_EQUALITY_HPP
#define DAMN_CORE_MATH_EQUALITY_HPP


#include <string>
#include <cstddef>

namespace damn::core
{

struct EqualityReport
{
    auto operator()() const { return eq; }
    operator bool() const { return eq; }
    operator std::string() const { return reason; }
    auto& what() const { return reason; }
    auto& why() const { return reason; }
    bool operator==(bool b) const { return eq == b; }
    bool const eq            = true;
    std::string const reason = "==";
    std::size_t idx          = 0;
};

} // namespace damn::core

#endif /* DAMN_CORE_MATH_EQUALITY_HPP */
