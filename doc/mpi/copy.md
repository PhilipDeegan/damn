# Copies over MPI (`mpi/copy.hpp`)

`damn::mpi::copy` does the same copies as `damn/amr/copy.hpp` (`doc/amr/copy.md`), between
patches of any rank. It is collective over the ranks holding the level's patches.

```cpp
damn::mpi::copy(damn::amr::fill_ghosts, neighbours, {"Bx"}, level_data);
```

The resources are reached through `access`: `amr::by_name`, or accessors one per pair, as in
`damn::amr::copy`. Given names, a field has the same name on both sides.

## Messages

Every rank plans the same pieces from the same global patch list. So a message holds the pieces'
data only, in plan order, with no headers: one message per peer rank and copy. A message that
does not match the receiver's plan is an error.

## Steps

| Step | What |
|---|---|
| 1. send | outgoing pieces packed per peer and sent (`MPI_Isend`) |
| 2. local | local pieces: ghosts and leaving items copied, sums and maxes read into the buffer |
| 3. receive | per peer with pieces for this rank: probed (item messages vary in size), received, unpacked; sums kept in the received buffers |
| 4. finish | local maxes written, sums folded with the received ones in a fixed order (`damn::amr::detail::fold_sums`), sends waited for |

Every source is read (steps 1 and 2) before anything is written (steps 3 and 4), so sums and
maxes use the values from before the copy. The fixed fold order makes sums the same bits
whatever order the messages arrive in.

## On the device

Resources on the device (`doc/amr/device_copy.md`) never go through the host:

| Step | What |
|---|---|
| 1. receive | a receive posted per peer, into the plan's device receive buffer: a field message's size is the plan's |
| 2. send | the outgoing pieces packed into the device send buffer (one launch), a send per peer |
| 3. wait | for the receives; a message of another size than the plan's is an error |
| 4. finish | ghosts assigned, sums and maxes folded, local and received pieces in one launch; sends waited for |

MPI is handed device pointers, so it must be GPU aware (`MPIX_Query_cuda_support`,
`MPIX_Query_rocm_support`): a copy with messages throws otherwise. Messages are the host path's
bytes. Leaving items move as columns, a counts message before their blocks:
`doc/amr/device_items.md`.

## In flight

`damn::mpi::Copy` is `damn::mpi::copy` spread over the level's work, so messages go while
patches still compute (`damn::mpi::copy` is a `Copy` finished at once):

```cpp
damn::mpi::Copy c{damn::amr::sum_borders, n, pairs, damn::amr::by_name, ld};
pool.patch_exec(n_patches, make_view, work,
                damn::core::ThreadPool::PatchHooks{.order   = c.remote_first(),
                                                   .on_done = [&](auto i) { c.done(i); },
                                                   .idle    = [&] { c.poll(); }});
c.finish();
```

| Call | What |
|---|---|
| `remote_first()` | the local patches feeding other ranks first, then the others, each in index order: a dispatch order |
| `done(i)` | local patch `i`'s sources are final: each peer whose pieces all come from done patches is packed and sent (step 1 for that peer, the same message as `copy`'s) |
| `poll()` | the messages that have arrived are received into the plan's buffers; nothing is written to a patch |
| `finish()` | the peers not sent yet are sent, then steps 2 to 4 |

Nothing is written to a patch before `finish()`, and sums and maxes fold in the same fixed
order, so the result is the same bits as `copy`. A patch must not change its sources after
`done(i)`, nor any patch its destinations before `finish()`.

The schedule (per peer its pieces, per local patch the peers it feeds) is built once per plan
and kept in it (`plan.in_flight`), with the per copy state, so a step does not allocate. One
`Copy` is in flight per rank at a time: messages of every kind share one tag (#57), and a second
`Copy` throws. MPI calls come from one thread (`MPI_THREAD_FUNNELED`), the one dispatching work.

On the device, one launch packs every piece, so `done` and `poll` do nothing and `finish` runs the
whole device copy.

`simulate_latency(fn)`, for benchmarks on one node, where shared memory delivers at once: a
message from `peer` counts as received `fn(peer, bytes)` seconds after this rank first sees it
(probed). `poll` leaves it until then, `finish` sleeps until then.
