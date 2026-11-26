# Regridding a level

`inc/damn/amr/regrid.hpp` moves a level's data from one decomposition (boxes and owners) to
another of the same cells: a rebalance (new owners), for example, or the same cells chopped
differently.

```cpp
auto map = make_regrid_map(h.levels[ln], new_level, rank);
ld       = regrid(map, std::move(ld), make_patch_data); // damn::mpi::regrid over ranks
set_level(h, new_level);
n = make_neighbours(...);                                // then refill ghosts
```

`make_regrid_map` takes the level's periodic images as `shifts` (`periodic_shifts`, the domain
itself excluded), so data moves across the periodic boundary too.

## What a new patch gets

A new patch with the same box and owner as an old one takes its data as is: moved, ghosts
included. Every rank finds the same kept patches.

Otherwise new patches are made (`make(patch)`) and get their domain from the old patches
overlapping or touching them, periodic images included:

| Resource | Taken |
|---|---|
| `BoxData` | the values on the domain points both share: `domain_box(resource, box)` of either. This includes, say, a primal node on the border of an old patch next to the new one, without shared cells |
| `CellItemData` | the items whose cell is in the shared cells: domain items only, since ghost items are outside every old domain |

This is done for every resource of those kinds (`for_each_resource`); other resources are left
as made. Ghosts are not filled: run `copy` with `fill_ghosts` (and the item ghosts) on the new
level afterwards.

## Valid old ghosts (`RegridGhosts`)

When the old ghosts are valid (a rebalance right after a step, say, with the ghosts as the step
left them), the resources named in `RegridGhosts` take their ghosts along too:

- a **field** resource gets its new ghost nodes from an old domain where one holds the node,
  otherwise from an old patch's ghosts (the level border and physical ghosts, which no old domain
  holds);
- an **item** resource gets its items in the ghost cells of both patches (at its own width). For
  level ghost items, a cell may be in two new patches' ghost cells: the caller keeps the copy of
  the owner.

`reach` is the widest ghost width among them, in cells; transfers are then listed between
patches within reach of each other.

What nothing covers is left as made (NaN in a debug build): the caller must refine or fill it.

## The map and messages

A `RegridMap` lists, from one rank's point of view, the transfers: what of one old patch (or a
periodic image of it) goes to one new patch. Both ranks of a pair list the transfers between
them in the same order (old level order, then new level order), so messages need no description
of their content. Within a transfer the points also come in an order both sides agree on: the
domain points both share; with valid ghosts, then the new ghost nodes the old domain holds; then
those the old patch's ghosts hold and no old domain does.

A transfer of items with no cells still writes a zero count, as both sides have the transfer.

## Memory

The old data is consumed. Old patches are visited in level order (along the curve the partition
follows), each freed once its data is copied or packed; a new patch is made when its first data
arrives. The peak is about the old level, not the old and the new together.

A kept old patch's data is already in its new patch, so its transfers (border points of the new
patches touching it) read from there.
