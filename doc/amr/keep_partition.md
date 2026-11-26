# Partitioning by distance

`inc/damn/amr/keep_partition.hpp` partitions for regrids and for the finer levels of a
rebalance, keeping data where it already is:

- data that stays on its rank costs nothing to move;
- a fine patch on the rank of the coarse cells under it refines and synchronizes without
  messages.

The curve of [partition.md](partition.md) balances each level on its own, so most of the coarse
cells under a fine patch end up on other ranks; these functions avoid that (see
[decompose.md](decompose.md) for the partitions built on them). They mirror `plamn/amr.py`'s
`Chopper` (partition `keep`, spill), their proof of concept.

## `keep(boxes, old, coarser, ratio, fresh)`

Each box (a level's cells) is cut into pieces with an owner, in this order of preference:

1. where a patch of `old` (the level being replaced) holds the cells: its owner, so the data
   stays;
2. elsewhere, where a patch of `coarser` (the next coarser level, refined by `ratio`) lies under
   them: its owner, so the cells refine from there;
3. the rest by `fresh(rest)`, which returns `OwnedBox`es covering those boxes exactly once (e.g.
   chopped, then partitioned along the curve). `keep` throws if they do not cover the rest's
   cells.

Every rank computes the same pieces.

## `owned_boxes(pieces, n_ranks)`

Per rank, its pieces coalesced (`coalesce` from `partition.hpp`): the boxes to chop into
patches.

## `spill(boxes, ranks, weights, topology, tol)`

The ranks again, with no rank above `(1 + tol)` times its share of the weight. A rank's share
comes from its node's share (`node_share_capacity`).

An overloaded rank, the most loaded first, hands over patches until it is within its limit:

- the patch handed over is the heaviest that fits in its excess, otherwise its lightest (ties:
  the lower index);
- it goes to the least loaded rank with room, the closest first: the same memory domain, then
  the same node, then any node (`Distance`);
- a rank keeps its last patch, and stops when no rank has room (patch granularity).

The result is deterministic.
