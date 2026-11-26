# Kelvin lattice sites

`inc/damn/amr/kelvin/sites.hpp` places a Kelvin (body centred) lattice's sites on damn's cells,
with their faces, in any dimension.

## Keys

A site's key is its position in `h` units, `h` being half a cell. Either every component is
even (a corner site) or every one is odd (a body site):

| Site | Key of cell `c` |
|---|---|
| corner | `2 c` |
| body | `2 c + 1` |

So each cell holds one site of each sublattice.

## Faces

A site has `2^dim + 2 dim` faces, indexed `k` in this order:

1. **diagonal faces** (`2^dim`): to the other sublattice, a step of `+-1` in every direction, in
   lexicographic order of the signs, `(-, -, -)` first;
2. **axial faces** (`2 dim`): to the same sublattice, a step of `+-2` along one direction, in
   the order `-x, +x, -y, +y, ...`.

In 3D these are the 8 hexagons and 6 squares of a truncated octahedron.

The neighbour across face `k` (its cell and site) is never more than one cell away. `mirror(k)`
is the same face seen from the other side.

## Domains: `KeyBox`

A `KeyBox` is a domain's sites: keys `lower..upper` per direction, each direction periodic or
bounded.

- A **periodic** direction's keys are whole cells: `lower` even, `upper` odd (`check` throws
  otherwise). Keys past its ends are periodic images, and `has` them.
- A **bounded** direction's keys need not be whole cells: with keys `-N..N`, say, one sublattice
  has a layer more than the other. Its cells are those holding a site, and keys past its ends are
  not in the domain.
