# Lattices

What a hierarchy's cells hold, given as a tag type: `Hierarchy<dim, Lattice>`.

| Lattice | Sites |
|---|---|
| `Cartesian` | the default: boxes of cells, nothing more said |
| `Kelvin` | a body centred lattice on the same boxes, two sites per cell |

## Kelvin

A body centred cubic (BCC) lattice: its Voronoi cells are Kelvin cells, truncated octahedra in
3D. Each cell holds two sites:

- the **corner**: the cell's lower corner;
- the **body**: the cell's centre.

Each cell owns one site of each kind, so patches share no site. A site's `2^dim` diagonal (hex)
neighbours and `2 dim` axial (square) neighbours all lie within one cell of it.

### Odd refinement ratios

Refinement ratios must be odd. With ratio `r`, an odd ratio keeps the two sublattices apart:

| Coarse site | Fine site |
|---|---|
| corner `i` | corner `r i` |
| body `i` | body `r i + (r - 1) / 2` |

With an even ratio, a coarse body would land on fine corners.

## Generic code

Generic code takes any lattice (`template<std::size_t dim, typename Lattice>`). What differs is
overloaded: the lattice's own overload is constrained by its concept (`KelvinLattice`), and
`Cartesian`'s is unconstrained and declared last.
