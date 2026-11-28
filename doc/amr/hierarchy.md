# Patch hierarchy

A patch hierarchy as plain data, with free functions over it. Nothing here communicates: ranks
are only numbers (`damn/mpi` does the communication).

## Data

| Type | Holds |
|---|---|
| `Hierarchy` | level 0's domain, mesh size, origin, periodicity, refinement ratio, the levels; its `Lattice` (`lattice.hpp`) is a tag: `Cartesian` by default, or `Kelvin` |
| `Level` | every patch of the level, whichever rank owns it; the same on every rank |
| `Patch` | a cell box in its level's index space, the owner rank, an id unique on the level |

Level `ln`'s index space is level 0's refined by `ratio^ln`. Boxes are cell boxes.

`set_level` replaces a level and removes the finer ones, since they were built on the replaced
one.

## Chopping: `chop_evenly(box, size, granularity)`

`chop` cuts a box into boxes of at most `size` cells per direction, which can leave a thin box at
the end. `chop_evenly` instead shares the cells evenly, so no box is left with a few cells (SAMRAI's
chop on one rank does the same).

Per direction:

1. the box is cut into the count of boxes nearest `width / size`; more only if a box would be
   wider than `chop_size_slack * size` (`size` is a hint, not a bound);
2. the cells are shared evenly in units of `granularity` cells from the box's lower corner. With
   the refinement ratio as granularity, refined boxes are made of whole coarser cells; `size` and
   its slack are rounded to whole units;
3. the leftover units go one each to the first boxes.

| size | granularity | cells | `chop_evenly` | `chop` |
|---|---|---|---|---|
| 20 | 1 | 100 | 20 x 5 | 20 x 5 |
| 20 | 1 | 45 | 23, 22 | 20, 20, 5 |
| 20 | 1 | 24 | 24 | 20, 4 |
| 20 | 1 | 26 | 13, 13 | 20, 6 |
| 60 | 2 | 122 | 62, 60 | 60, 60, 2 |

## Ghost cell ownership: `owned_elsewhere(h, ln, box, width)`

Returns the cells of `box` grown by `width` (a patch's ghost box, in level `ln`'s index space) that
another patch of level `ln` owns, periodic images included, as boxes of cells along the last
direction.

A cell's owner is, among the patch images holding it in their box grown by `width`:

- the nearest one: the L1 distance from the cell to the image's box;
- on a tie, the first in box order (`core::lexicographic_less`).

Both criteria are translation invariant and computed the same on every rank, so every holder
agrees on the owner.

A level ghost particle is held by its cell's owner only. Nearest rather than first means it stays
in its holder's ghost box the longest: for example a cell diagonal to one patch's corner and next
to another patch's side belongs to the latter.
