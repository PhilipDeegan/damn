# Patch costs

What a patch costs to advance, for partitioning (`decompose.hpp`). Partitioning balances a
`Weigh`: per level (its number) and box, a weight.

Every rank computes the same costs from the same counts: the counts are global, summed over the
ranks by the caller.

## `CostModel{item, cell, ghost_cell, patch, reach}`

A patch's cost from its box and a count in it. The count is whatever the model counts:
particles, a cell's refined cells, ...

```
cost = item * count + cell * cells + ghost_cell * ghosts + patch
```

where `ghosts` is the number of cells within `reach` of the box, outside it.

The sum is evaluated as nested fused multiply adds,
`fma(item, count, fma(cell, cells, fma(ghost_cell, ghosts, patch)))`. Each is rounded once
(IEEE), so the result is the same double whatever the target (`-march`) or compiler, and python's
`math.fma` gives it too: the same decomposition everywhere. Left to the compiler, `a*b + c*d`
fuses either way round, or not at all.

The default weighs the count alone. `measured_hybrid()` gives the per term times of a hybrid PIC
step (item: a particle), measured on one rank (3D, interpolation order 1, ghost reach 4) and
fitted to the busy time per patch of a decomposition log.

## `CostMap{box, block, counts}`

Counts on a grid of `block` cells from `box.lower`, x fastest; the last block of each direction
is clipped to the box. A box's count is summed over the blocks it overlaps, each block
contributing its count times the share of its cells inside the box.

It is for callers that do not hold the counted things themselves (an emulation, a decomposition
made before a run); a run counts its own.

## Building a `Weigh`

| Function | Gives |
|---|---|
| `weigh_by(model, counts)` | per level and box, the model's cost of `counts(ln, boxes)` |
| `counts_of(maps)` | a `Weigh` of per level `CostMap` counts: `weigh_by`'s input |
