# VTKHDF output (`hdf5/vtkhdf.hpp`)

VTK's OverlappingAMR in HDF5 (VTKHDF 2.2), as a time series: one file, a step per dump. Each
level holds its boxes, the point data on their primal nodes and, optionally, cell data on their
cells (centres). On a Kelvin lattice the corner sites are the points and the body sites the cells.

## Writing

| Function | What |
|---|---|
| `init(file, VtkInit{n_components, max_levels, spacing, chunk_bytes, ratio, n_cell_components})` | the groups, datasets and attributes of every level up to `max_levels`, if not there yet; `CellData/data` only with `n_cell_components` |
| `append_step(file, time)` | a step: its time, `NSteps + 1` |
| `append_level(file, VtkLevel{ln, level, rank, n_components, data})` | collective: level `ln` at the last step. Its boxes, their offsets, and each rank's rows at its place, one write per rank. A level without patches (not there yet) appends an empty entry. Its cell rows (`cell_data`) too if the file has cell data |

`VtkInit`: `n_components` columns per point, `max_levels` levels the file holds, `spacing` level
0's cell widths (a level's is level 0's / `ratio^level`), `n_cell_components` columns per cell
(0: no cell data).

## Rows

| Function | Rows of a box |
|---|---|
| `points_of(box)` | its primal nodes: cells + 1 per direction |
| `rows_of(box)` | its points, `x_times<dim>` times: in 1D and 2D the box is a 3D view, its block of points repeated for each dummy direction's two nodes |
| `cell_rows_of(box)` | its cells (a dummy direction has one) |

- Points are in Fortran order (x fastest), one column per component: vectors x y z, tensors
  xx xy xz yy yz zz.
- The caller's data is in its patches' order (level order).
- Boxes are rank major: every rank's patches one after another, by rank, each owner's in level
  order.
- AMRBox rows: the cells, lower and upper per direction (upper inclusive), 0 for directions past
  `dim`.

## Collectives

Every rank calls every resize with the same sizes. Rank 0 writes the steps' entries and the
boxes; each rank writes its own rows.
