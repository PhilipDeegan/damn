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
bytes. Leaving items (Append) take the host path.
