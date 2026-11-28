# Refine and coarsen over MPI (`mpi/inter_level.hpp`)

`damn::mpi::refine` and `damn::mpi::coarsen` do the same transfers as
`damn/amr/inter_level.hpp` (`doc/amr/inter_level.md`), between patches of any rank. They are
collective over the ranks holding either level's patches.

```cpp
damn::mpi::refine(op, LevelRegion::LevelGhosts, map, pairs, coarse_ld, fine_ld, alpha);
damn::mpi::coarsen(op, map, pairs, fine_ld, coarse_ld);
```

The resources are reached through `access`, as for `damn::amr::refine`: `amr::by_name`, or
accessors one per pair.

## Messages

Every rank plans the same pieces in the same order: fine patches, then their links, in level
order. So a message holds the pieces' data only: one message per peer rank and transfer. Each
transfer kind has its own tag (`copy_tag`, `regrid_tag`, `inter_level_tag`).

| Transfer | Coarse ranks | Fine ranks |
|---|---|---|
| refine | pack and send (`MPI_Isend`) the coarse values other ranks' fine patches need; local pieces go directly | receive (`MPI_Probe`: item messages vary in size) and unpack, then `op` fills the fine field points from the gathered coarse values |
| coarsen | receive | compute (`op`) on their fine patches, then send, or assign directly when the coarse patch is local |
