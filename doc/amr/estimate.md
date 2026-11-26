# Step time estimate

A decomposition's step time, estimated without data. It is used to choose between
decompositions (`decompose.hpp`'s partitions, spill tolerances, patch sizes) by what a run would
take.

It is an estimate, not a prediction: compute and exchange do not overlap, and every exchange of a
step is folded into `bytes_per_cell`. That is good enough to rank decompositions.

## The model: `estimate(h, topology, reach, weights, costs[, threads, detached])`

Per level and rank, two terms:

- **compute**: the rank's patches' weights, in seconds (`cost.hpp`'s `CostModel`). Without
  weights for a level, a patch weighs its cell count.
- **exchange**: per `Distance`, a latency per message plus a cost per cell overlapped. A message
  is a pair of patches overlapping:
  - on the same level: another patch's cells, or its periodic image's, within `reach` of a patch;
  - with the coarser level: the coarser patches under a patch grown by `reach`, coarsened.

A level takes its slowest rank's compute + exchange, and level `l` runs `ratio^l` times per coarse
step:

```
step = sum_l ratio^l max_r (compute[r] + exchange[r])
```

## Exchange costs

`ExchangeCosts{latency, per_cell}` holds, per `Distance`, seconds a message and seconds a cell.

`exchange_costs(node_latency, node_s_per_byte, bytes_per_cell)` takes the `Node` distance's costs
from a measurement and sets the others relative to it. The factors are placeholders until
measured:

| Distance | Self | Local | Numa | Node | Socket | Remote |
|---|---|---|---|---|---|---|
| factor | 0 | 0 | 0.3 | 1 | 2 | 10 |

`bootstrapped_exchange_costs(bytes_per_cell)` reads this machine's bootstrap
(`etc/bootstrap/mpi/exchange_*`: `Node`, and `Socket` where measured), else defaults to a shared
memory exchange of about 1 us latency and 1e-10 s a byte. `bytes_per_cell` is what a model
exchanges per ghost cell.

## Threads

`Threads{per_rank, main_busy, efficiency}` models a rank's threads:

- the rank's patches are split into tiles (`TileSet`); a patch's cost is shared by its tiles'
  cells;
- tiles go largest first, each to the least loaded thread; the main thread has only
  `1 - main_busy` of its time for tiles (the rest is MPI);
- the rank's compute is its busiest thread's time over `efficiency` (shared bandwidth).

With one thread, a rank's compute is its patches' sum over `1 - main_busy`.

Fewer ranks with more threads means fewer exchanges, and the imbalance of whole patches within a
rank is gone, at the cost of the threads' efficiency.

## Detached level 0

`Detached{near_band}` gives `Estimate::detached`: the step with level 0's far patches run as
background work on a rank's spare threads. Far patches are those beyond `near_band` cells of
level 1; they are needed only by the end of the coarse step. The estimate is the maximum over
ranks of the larger of its far part and the rest.

This applies only with threads (`per_rank > 1`). With one thread per rank, far level 0 stays in
level 0's phase (collectives and dumps need every rank there), which is `Estimate::step`.

## Choosing: `choose(h, from, boxes, candidates, reach, costs[, threads])`

Each candidate (a `Decompose`: partition, spill, sizes, ...) decomposes levels `from`,
`from + 1`, ...; `h`'s levels under `from` are kept. Each decomposition is weighed by its own
`weigh` and estimated, and the cheapest is kept (ties: the first).
