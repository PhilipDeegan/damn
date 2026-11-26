# Execution model (`exec/execution_model.hpp`)

`ExecutionModel` says what one rank runs on, and how many of its threads are in use.

## Members

| Member | Meaning |
|---|---|
| `mpi_ranks` | 0: MPI not initialised (one process, no messages); else the run's ranks. `damn/mpi/execution_model.hpp` sets it from MPI |
| `usable_threads` | the process's threads, per rank when `mpi_ranks > 0` (each rank has its own model). 1, the default, is the main thread only |
| `max_scheduled` | threads in use at once are at most this times `usable_threads`, and at least 1. 1.2 by default |
| `threads_used` | in use now: the main thread (1) and every lease's |

`max_scheduled` above 1 oversubscribes a little: some oversubscription hides waits, more only
contends. What suits depends on the work and the machine.

| Query | True when |
|---|---|
| `mpi()` | ranks exchange messages |
| `threaded()` | more than the main thread is usable |
| `mpi_only()` | ranks of one thread each |

## Leases

`use_threads(n)` leases up to `n` more threads, whatever is free when called, until the
`ThreadLease` is destroyed. It can be none: the caller then runs the work itself.

Pools anywhere downstream (`LeasedPool`) size themselves from their lease, so pools
in different contexts share one budget and the rank never runs more than `schedulable()` threads.
