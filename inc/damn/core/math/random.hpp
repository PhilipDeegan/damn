#ifndef DAMN_CORE_MATH_RANDOM_HPP
#define DAMN_CORE_MATH_RANDOM_HPP

#include "mkn/kul/random.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

// a reproducible random stream per cell, whatever the patches and ranks: doc/core/math/random.md

namespace damn::core
{
// a pure function of the cell, never of its patch; negative indices sign extended: -2 != 2
template<std::size_t dim>
constexpr std::uint64_t cell_key(std::uint64_t const seed, std::uint64_t const populationKey,
                                 int const level, std::array<int, dim> const& amrCell)
{
    auto key = mkn::kul::hash_combine(mkn::kul::hash_combine(seed, populationKey),
                                      static_cast<std::uint64_t>(level));
    for (auto const index : amrCell)
        key = mkn::kul::hash_combine(key,
                                     static_cast<std::uint64_t>(static_cast<std::int64_t>(index)));
    return key;
}

} // namespace damn::core

#endif
