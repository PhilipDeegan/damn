# Clustering

`inc/damn/amr/clustering.hpp` makes the boxes of a new level `ln + 1` from the cells of level
`ln` tagged for refinement, by tiles, as SAMRAI's `TileClustering` does.

## Steps

1. Each tagged cell is grown by `tag_buffer` cells.
2. The tiles of the grown cells are found: the level's cells cut in `tile_size` boxes from cell
   0, periodic images wrapped into the domain, clipped to it.
3. Each tile keeps its cells properly nested in level `ln`: those with every cell within
   `nesting_buffer` on a patch of level `ln` (or past a physical, non periodic, domain
   boundary).
4. The tiles left are coalesced: two boxes whose union is a box become one, until none do.
5. The boxes are refined into level `ln + 1`'s index space.

## Options

| Option | Meaning (in level `ln` cells) |
|---|---|
| `tag_buffer` | how far each tagged cell is grown |
| `nesting_buffer` | how far inside level `ln`'s patches the new level's cells stay |
| `tile_size` | the tile grid's box size, from cell 0 |

## Ranks

`tagged_tiles` (steps 1 and 2) works on one rank's tagged cells. `cluster` (steps 3 to 5) works
on every rank's tiles, so every rank gets the same boxes: `damn/mpi/clustering.hpp` gathers the
tiles over the ranks in between.
