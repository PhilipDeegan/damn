# damn overview

damn (Distributed Adaptive Mesh Numerics) is the data layer under plamn: index space, storage,
execution, a patch hierarchy and its communication. It moves data and does not interpret it:
what a patch holds (fields, particles, populations, ...) is the user's type, and the arithmetic
of anything physical (refinement operators, centerings, boundary values) is the user's too.

This is a work in progress: the headers' leading comments are the reference, this page is only
a map of them.

## Layout

| Path | What |
|------|------|
| `inc/damn/core/geometry` | `Point`, `Box`, ranges and box spans: index space |
| `inc/damn/core/data` | n-dimensional arrays (`ndarray`), tile sets over boxes |
| `inc/damn/core/memory` | spans and allocation (host, device, unified) |
| `inc/damn/core/exec` | kernels (CPU loops, or mkn.gpu launches), the thread pools |
| `inc/damn/core/def` | build configuration, GPU/AVX/thrust detection, phlop scopes |
| `inc/damn/amr` | the hierarchy and everything done on it, single process |
| `inc/damn/mpi` | the same operations over MPI ranks |
| `inc/damn/hdf5` | HDF5 through HighFive: checkpoints, level tables, VTKHDF |
| `etc/bootstrap` | probes of the build's toolchain (MPI, HDF5, CUDA features) |
| `res/mkn` | mkn settings files: compilers, MPI, GPU backends, sanitizers |

## The hierarchy (`damn/amr`)

- **Hierarchy, Level, Patch** (`hierarchy.hpp`): plain data, the same on every rank. A level is
  every patch's box and owner; level `n`'s index space is level 0's refined by `ratio^n`.
- **LevelData** (`level_data.hpp`): one rank's patch data on one level. Generic code reaches a
  patch's resources only through the user's `for_each_resource`, or through accessors the user
  hands it.
- **Copies** (`copy.hpp`, `neighbours.hpp`): same level exchanges along neighbour links: ghost
  fills, border sums and maxes (deposits, reproducible bit for bit), leaving particles.
- **Inter level** (`inter_level.hpp`, `level_map.hpp`): refine and coarsen between consecutive
  levels. damn works out what goes where; the user's operator does the numerics.
- **Time refinement** (`time_refinement.hpp`): subcycled advance over the levels through user
  hooks (advance, prepare, first/last substep, synchronize).
- **Tagging to boxes** (`clustering.hpp`): tagged cells to a new level's boxes by tiles, nested
  in the level below.
- **Load balance and regrid** (`partition.hpp`, `keep_partition.hpp`, `regrid.hpp`): patches to
  ranks along a Hilbert curve by weight, then data moved from one decomposition to another.
- **Decomposition** (`decompose.hpp`, `cost.hpp`, `estimate.hpp`): every level's patches and
  owners from a cost model (particles, cells, ghosts, patches), by strategy (`levels`, or
  `finest`: the finest level first, coarser cells under it with its owners), and an estimate of
  a step's time to compare candidates; the same code in a run and out of one (python bindings).
- **Physical boundaries** (`physical_boundary.hpp`): which ghost regions lie past a non periodic
  boundary; what fills them is the model's.
- **Metrics** (`metrics.hpp`): per level imbalance and communication volume of a decomposition.

`damn/mpi` mirrors these (copy, inter level, clustering, regrid) as collectives: every rank plans
the same exchanges from the same global patch list, so messages carry data only.

## Execution

Kernels run as plain loops on the host or as mkn.gpu launches (CUDA/HIP) depending on the
build. `ThreadPool` maps patches to pools and tiles to a pool's threads;
`-DDAMN_THREAD_POOL_SYNC=1` runs every task on the calling thread, for serial timings, hardware
counters and debugging.

## Instrumentation

Scopes (`DAMN_LOG_SCOPE(level, "name")`) are phlop scope timers. What each level records (time,
memory, hardware counters, I/O) is chosen at compile time (`DAMN_LOG_SCOPE_FLAGS_<level>`) and
enabled at run time by environment variables; see `damn/core/def/phlop.hpp`. phlop's own
command line (`python3 -m phlop.timing.scope_timer`) reads the output.

Debug checks: `DAMN_DEBUG_DO(...)` is on unless `NDEBUG`; `DAMN_SUPER_DEBUG_DO(...)` (whole level
invariants, expensive) only with `-DDAMN_SUPER_DEBUG`.

## Building and testing

Profiles are in `mkn.yaml`:

| Profile | What |
|---------|------|
| `damn_core` | the core library (header only but for phlop's scope timer) |
| `damn_mpi` | MPI communication, needs an MPI compiler (`-x res/mkn/mpi`) |
| `damn_hdf5` | HDF5 over `damn_mpi`, needs a parallel HDF5 |
| `test_damn_core`, `test_damn_mpi`, `test_damn_hdf5` | the tests of each |
| `tidy`, `iwyu`, `iwyu_fix` | clang-tidy and include-what-you-use |

```sh
mkn clean build test -p test_damn_core -dKO -t 4 -x res/mkn/thrust
mkn clean build test -p test_damn_mpi  -dKO -t 4 -x res/mkn/mpi
```

`res/ci/local.sh` runs what CI runs, locally.
