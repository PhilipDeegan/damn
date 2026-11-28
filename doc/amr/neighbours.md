# Neighbours

`Neighbours` holds the patches of a level near each of a rank's patches, periodic images
included. It is purely geometric: built once per regrid (`make_neighbours`), after which
`copy.hpp` copies data along its links.

## Links

A `Link` is another patch (or a periodic image of the same one), the shift to its image, and how
far its data is (`Distance`).

Per local patch, in `LevelData` order, the neighbours are every patch whose box grown by `reach`
meets the local patch's box grown by `reach`. `reach` is the most cells any resource reaches
outside its patch: its ghosts, plus one for primal nodes.

`remote` holds other ranks' patches near this rank's, in level order, with all their links. This
rank sends them what the links to its patches give; the receivers decide which image of a point
fills a ghost.

Every rank makes the same links for a pair of patches, in level order, so their pieces come in
the same order.

## Plans

What moves where for a given set of resources is planned on first use and kept in the
neighbours (`Neighbours::plans`, by region and resource pairs). A regrid builds new `Neighbours`
and so drops the plans.

| Region | From | To |
|---|---|---|
| `Ghosts` | neighbours' domain points | this patch's ghost points |
| `Borders` | neighbours' ghost points | the same points of this patch (ghost boxes overlap) |
| `Leaving` | neighbours' items outside their domain | this patch's domain, where their cell is this patch's |

A `Plan` is what goes where for a region and set of resources, from one rank's point of view:

- `pieces`: into this rank's patches, from its own or another rank's;
- `outgoing`: from this rank's patches into another rank's.

Both ranks of a transfer make the same pieces in the same order, from the same global patch
list, so messages need no description of their content. What follows from the pieces alone is
made with the plan (`copy.hpp`'s `plan_for`), so a copy only moves values: per sending rank the
indices of its pieces, and for `Borders` the fold order (pieces by destination and resource pair,
then by source image in the destination's space) and its groups.
