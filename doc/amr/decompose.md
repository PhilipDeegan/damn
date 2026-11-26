# Decomposition

A hierarchy's decomposition: each level's boxes (its cells, from clustering or given) are chopped
into patches, and the patches are given ranks. A run and a decomposition made without one
(plamn's bindings) take the same code path, so both give the same patches and owners.

Every rank computes the same levels from the same weights. `weigh` may be collective: it is
called in the same order on every rank.

## Inputs (`Decompose`)

| Member | Meaning |
|---|---|
| `topology` | the ranks and where they are |
| `weigh` | per level and box, what to balance: a `CostModel` over counts (`cost.hpp`), or the caller's own |
| `patch_size` | the largest patch, in the level's cells, on every level unless `level_size` is given |
| `level_size` | a patch size per level; the last one applies to the finer levels |
| `under_size` | per level, the patch size for the cells under the next finer level, chopped apart from the rest so a coarse patch sits under one fine patch; zeros: none |
| `partition` | `levels`, `keep`, `finest` or `aligned` (below) |
| `spill` | `keep`, `finest`, `aligned`: `spill`'s tolerance; negative: no spill |
| `block` | `cut_partition`'s block, cells per direction |
| `cut` | cut patches where a rank's share ends (several ranks); false: patches stay whole |
| `near_band` | `finest` in pieces: level 0's cells within this many cells of level 1 go to level 1's owners |

A refined level's sizes are rounded down to a multiple of the ratio, at least the ratio, so its
patches are made of whole coarser cells.

## One level: `decompose_level(h, ln, boxes, in[, finer])`

Level `ln` from its boxes. `h` holds the coarser levels and, for `keep`, the level being
replaced. `finer` is the next finer level's boxes, if known (for `under_size`).

- **levels**: boxes chopped evenly (`chop_evenly`). On one rank, patches are partitioned whole
  along the curve. On several ranks, patches are cut where a rank's share ends inside one
  (`cut_partition`, blocks of `block` cells, refined levels in whole coarser cells), so every
  rank gets work whatever the patch size. With `cut` false they are partitioned whole.
- **keep**: cells the replaced level held go to their owner, otherwise to the owner of the
  coarser patch under them; the rest is chopped and partitioned along the curve
  (`keep_partition.hpp`). Each rank's pieces are coalesced and chopped, then spilled when
  `spill >= 0`.
- **finest** / **aligned** with a level alone (the finer levels unknown): to the owners of the
  coarser cells under it, spilled. Level 0 is done as `levels`.

## Several levels: `decompose(h, from, boxes, in)`

Levels `from, from + 1, ...` from `boxes[0], boxes[1], ...`. `levels` and `keep` go level by
level, each made on the ones before it. `finest` and `aligned` use every level at once.

### finest

The aim: a fine patch and the coarse cells under it on one rank, so refining and synchronizing
need no messages.

Cut, on several ranks (`cut` true): each level from finest to coarsest, in pieces. Cells under
the finer level go to the owners of the finer patches above them; otherwise, for the coarsest new
level when `from > 0`, to the owners of the fixed coarser cells under them. The rest is cut where
a rank's share ends. Each rank's pieces are coalesced and chopped, then spilled. A coarse patch is
never given whole to one rank for the fine cells above only part of it.

With `near_band`, level 0's cells within `near_band` of level 1 (what level 1's coarse step
reads, where its level ghosts split from) also go to level 1's owners. Level 0 beyond that band is
needed only at the end of the coarse step, so it is cut along the curve with the rest of level 0:
with one thread per rank it stays in level 0's phase (collectives, dumps); with more it can run
as background work, placed contiguously.

Whole patches (`cut` false, or one rank): the finest level is partitioned whole along the curve.
Each coarser level, finer to coarser, gives a patch to the owner of most of the fine cells above
it; patches with nothing above are partitioned along the curve among themselves. Then spill, if
`spill >= 0`.

With `from > 0` (a regrid of the finer levels, `from - 1` fixed): the coarsest new level's
patches with nothing above them (on the finest level alone, all of them) go to the owner of most
of the fixed cells under them.

### aligned

Level `from` along the curve (`from > 0`: on the fixed level under it). Each finer level, coarser
to finer, gives a patch to the owner of most of the coarse cells under it. Then spill, if
`spill >= 0`.
