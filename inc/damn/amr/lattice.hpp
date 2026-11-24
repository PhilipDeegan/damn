#ifndef DAMN_AMR_LATTICE_HPP
#define DAMN_AMR_LATTICE_HPP

// what a hierarchy's cells hold, as a tag: Hierarchy<dim, Lattice>
//
//   Cartesian: the default, boxes of cells and nothing more said
//   Kelvin:    a body centred lattice (BCC: Kelvin cells, truncated octahedra in 3D) on the
//              same boxes, two sites per cell: its corner (the cell's lower corner) and its body
//              (the cell's centre). Each cell owns one of each, so patches share no site, and a
//              site's 2^dim diagonal (hex) neighbours and 2 dim axial (square) neighbours are all
//              within one cell of it. Refinement ratios are odd: an odd ratio keeps the
//              sublattices (coarse corner i: fine corner r i, coarse body i: fine body
//              r i + (r - 1) / 2), an even one would land a coarse body on fine corners
//
// generic code takes any lattice (template<std::size_t dim, typename Lattice>); what differs
// is overloaded, the lattice's own constrained by its concept, Cartesian's unconstrained and last

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

// a Kelvin cell's two sites
enum class Site : std::uint8_t { corner, body };

} // namespace damn::amr

#endif // DAMN_AMR_LATTICE_HPP
