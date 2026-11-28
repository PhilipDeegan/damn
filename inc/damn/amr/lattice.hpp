#ifndef DAMN_AMR_LATTICE_HPP
#define DAMN_AMR_LATTICE_HPP

// what a hierarchy's cells hold, as a tag: doc/amr/lattice.md
// Kelvin ratios are odd: an even one would land a coarse body on fine corners
// what differs is overloaded: the lattice's own constrained by its concept, Cartesian's
// unconstrained and last

#include <concepts>
#include <cstdint>

namespace damn::amr
{
struct Cartesian
{
};

struct Kelvin
{
};

template<typename Lattice>
concept KelvinLattice = std::same_as<Lattice, Kelvin>;

namespace kelvin
{
    // a Kelvin cell's two sites
    enum class Site : std::uint8_t { corner, body };
} // namespace kelvin

} // namespace damn::amr

#endif // DAMN_AMR_LATTICE_HPP
