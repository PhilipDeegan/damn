# Decomposition metrics

What a hierarchy's decomposition costs, without data, for comparing decompositions (chopping,
partitioning) before a run. Exchanges are counted by `Distance` (`topology.hpp`).

## Per level: `metrics(h, topology, reach, weights)`

| Member | Meaning |
|---|---|
| `ghost_cells[d]` | same level: the cells of other patches (and periodic images) inside each patch's box grown by `reach`, whose owner is at distance `d` |
| `coarse_cells[d]` | levels > 0: the coarser level's cells under each patch's box grown by `reach` (coarsened), whose owner is at distance `d`: what refining its ghosts reads |
| `weight[r]` | rank `r`'s sum of the level's patch weights (cells if none are given) |
| `imbalance()` | the busiest rank's weight over the mean (1: balanced) |

`weights[ln]` holds one weight per patch of level `ln`; empty or missing means the patch's cells.
`reach` is in each level's cells.

`Self` counts a patch's own periodic images; `Local` counts the same rank's other patches.

## Moving between decompositions: `migration(from, to, topology)`

What moving from hierarchy `from` to `to` (a regrid, a rebalance) costs, without data, per level
of `to`, by `Distance` from the old owner to the new one:

| Member | Meaning |
|---|---|
| `moved[d]` | cells of a new patch the old level held (any patch), owned at distance `d` |
| `refined[d]` | cells of a new patch the old level did not hold, refined from the old coarser level's cells under them (coarsened), whose owner is at distance `d`. Level 0 has none: it always covers the domain |

`Self` and `Local` are free, since the data stays on its rank; the rest is copied between ranks.
