#ifndef DAMN_CORE_MATH_RANDOM_HPP
#define DAMN_CORE_MATH_RANDOM_HPP

#include "mkn/kul/random.hpp"

#include <array>
#include <cstdint>
#include <cstddef>

/* Tools to give every cell its own reproducible random stream.
 *
 * Seeded particle loading must not depend on how a level is cut into patches or spread over
 * MPI ranks. So the random numbers of a cell are derived from what identifies the cell, never
 * from the patch: (user seed, population, level, global AMR cell index) is hashed into a
 * 64-bit "cell key", and that key seeds a small generator used for that cell only.
 *
 *   seed ──┐
 *   pop  ──┼─ hash_combine chain ─> cell key ─> SplitMix64 ─> deltas, velocities of the cell
 *   level ─┤
 *   cell ──┘
 *
 * Everything here is plain 64-bit integer arithmetic, so it gives the same bits on every
 * platform and compiler (unsigned overflow wraps modulo 2^64 by definition in C++).
 */

namespace damn::core
{
/** @brief key of the random stream of one cell, a pure function of (seed, population, level,
 * global AMR cell index): it does not depend on how the level is cut into patches.
 *
 * - population: two populations with the same seed must not get the same particles;
 * - level: the same AMR index exists on every level but is a different physical cell;
 * - cell: global AMR index, the same whichever patch or rank holds the cell. Negative indices
 *   are sign-extended to 64 bits, so -2 and 2 give different keys.
 *
 * Combining order: seed, population, level, cell[0], cell[1], cell[2].
 * For a fixed prefix the last combine is bijective, so cells differing in their last index
 * never share a key; other collisions have probability ~2^-64.
 */
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
