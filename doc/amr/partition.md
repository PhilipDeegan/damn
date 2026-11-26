# Partitioning along the Hilbert curve

`inc/damn/amr/partition.hpp` balances load: it gives boxes (patches) to ranks along a Hilbert
space filling curve.

## Whole boxes: `partition`

```cpp
partition(boxes, weights, capacity); // each box's rank in [0, capacity.size())
partition(boxes, weights, n_ranks);  // the same, every rank's capacity 1
```

The curve is cut into one run per rank, rank `r`'s share of the weight being
`capacity[r] / sum(capacity)`.

- **capacity**: what each rank has for this work, e.g. its threads less those given to something
  else (analysis, diagnostics, a minimization); 0 for a rank that takes none.
- **weights**: whatever costs, e.g. cells (`cell_weights`), or particles (density above a cutoff
  times particles per cell). When nothing weighs, boxes are shared by count.

A box goes to the rank whose share of the total weight holds the middle of the box's weight
along the curve. So every rank ends within one box's weight of its share, and each rank's boxes
are a contiguous run of the curve. The curve keeps a rank's boxes together, so most neighbours
are on the same rank.

The partition is deterministic: every rank computes the same one from the same input weights.

## The curve

`hilbert_index` follows J. Skilling, "Programming the Hilbert curve", AIP Conf. Proc. 707, 381
(2004): consecutive indices are face neighbours. `curve_order` orders boxes by the index of
their centres (twice the centres, to stay on integers), with enough bits for the boxes' index
space.

## Cut boxes: `cut_partition`

```cpp
cut_partition(boxes, block, block_weights, capacity[, tol]); // or cut_partition(CutPartition{...})
equal_capacity(n_ranks);                                      // n_ranks ranks of the same capacity
```

This plays the role of SAMRAI's `CascadePartitioner`, but along the curve. With whole boxes the
ranks only get within a box's weight of their share, and patches can be large (tiled): a single
patch can weigh more than the imbalance allowed (`tol` of a share).

So boxes are cut where a rank's share ends inside them:

1. a box is cut into blocks (`blocks_of`): `block` cells per direction from its lower corner,
   the remainder in the last block of each direction (`block` to `2 * block - 1` cells, or the
   box's width if less);
2. a box that a rank's share ends in is split by its blocks along the curve, each block to the
   rank holding its middle;
3. the blocks of each rank are coalesced back into boxes (`coalesce`).

A box is cut only if more than `tol` of the smallest non zero share lies on both sides of a share's
end; otherwise it goes whole to the rank holding the middle of its weight. Each rank ends within
a block's weight of its share, plus `tol` of the smallest share for each end of its run left
uncut.

Weights are per block, in `blocks(boxes, block)` order (box after box), and the same on every
rank: e.g. particles counted per block on the rank that holds the box, then summed over the
ranks.

The result is a `Partition`: the boxes along the curve, and their ranks, non decreasing.
