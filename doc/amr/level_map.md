# Level map

A `LevelMap` records how the patches of two consecutive levels overlap, periodic images included.
Like `Neighbours`, it is geometric only, and is rebuilt on a regrid of either level.

## Links

| Member | For | Links to | Shifts in | Used for |
|---|---|---|---|---|
| `under[i]` | the i-th local fine patch | the coarse patches under its box grown by `reach` (fine cells) | the coarse index space | filling its level border ghosts, or a new patch, from coarse data |
| `over[i]` | the i-th local coarse patch | the fine patches over it | the fine index space | coarsening fine data onto it |
| `remote` | other ranks' fine patches over this rank's coarse patches, in level order | this rank's coarse patches only | the coarse index space | what this rank sends them (refine) or receives from them (coarsen) |

Every rank makes the same links for a pair of patches, in level order.

## Plans

`inter_level.hpp` refines and coarsens data along the map. It plans what goes where on first use
and keeps the plans in the map (`LevelMap::plans`), keyed by region, resource pairs and stencil.

A `LevelPlan` holds, from one rank's point of view, every target and piece with a side on this
rank, in the same order on every rank (fine patches in level order, their links in level order).
Messages therefore need no description of their content.

| Region | Direction | Points |
|---|---|---|
| `Interior` | refine | the fine patches' domain points (or cells), e.g. a new level |
| `LevelGhosts` | refine | fine ghost points (or cells) on no fine patch's domain |
| `Coarsened` | coarsen | the coarse domain points under the fine patches' domains |

A `LevelPair` names the coarse and the fine resource (usually the same name for fields). With
`coarse_new` set, refining reads `(1 - alpha) coarse + alpha coarse_new`: coarse data
interpolated in time.

`is_current(map, h)` says whether both levels still have the patches the map was built on.
