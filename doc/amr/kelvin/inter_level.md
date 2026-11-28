# Kelvin lattice sites between levels

`inc/damn/amr/kelvin/inter_level.hpp` maps a Kelvin lattice's sites ([sites.md](sites.md))
between levels at an odd ratio `r`. Each sublattice nests in itself, as a staggered Cartesian
grid does, so a site maps to sites of its own sublattice only. Even ratios do not nest (a coarse
body site would fall between fine body sites), so `check_ratio` throws for them.

## Nesting

| Sublattice | Position | Coarse site `c` is fine site |
|---|---|---|
| corner | cell corners (as primal nodes) | `r c` |
| body | cell centres | `r c + (r - 1) / 2` (r odd) |

## Functions

| Function | Gives |
|---|---|
| `coincident_fine(coarse cell, site, r)` | the fine cell whose site is the coarse one |
| `coincident_coarse(fine cell, site, r)` | the coarse cell whose site the fine one is, if any |
| `bracket(fine cell, site, r)` | per direction, the coarse cells `c` and `c + 1` of the site's sublattice around the fine site, which is at `c + num / r` (`num` 0: on `c`) |
| `refined_keys(keys, r)` | a domain's sites on the level `r` times finer |

A fine site's bracket is at most one coarse cell from the cell over it, so a refine stencil of 1
is enough.

In `refined_keys`, key `K` becomes the fine key `r K`, on its own sublattice. In a periodic
direction the last coarse cell becomes `r` fine cells, so the upper key gets `r - 1` more.

## Operators

What the values do is the model's: `damn::amr::refine` and `coarsen`
([../inter_level.md](../inter_level.md)) call its operators. These two are the plain ones, on a
`SiteArray` of a floating point `T`, or an array of them:

| Operator | Does |
|---|---|
| `LinearRefine` | a fine site from its bracket, multilinear (stencil 1) |
| `InjectCoarsen` | a coarse site from its coincident fine site |
