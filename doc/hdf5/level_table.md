# Level tables (`hdf5/level_table.hpp`)

A level's patches' rows in one dataset, as VTKHDF does: the patches one after another in the
level's patch order (every rank's), patch `k`'s rows from `offsets[k]` to `offsets[k + 1]`.

## Functions

| Function | What |
|---|---|
| `level_rows(level, rank, local_rows)` | collective: `LevelRows{level, rank, table}`, the table built from the row count of each of this rank's patches (in level order): every patch's offsets, the same on every rank |
| `write_level(file, group, rows)` | the level's boxes, ids, owners and the offsets |
| `read_level<dim>(file, group, number)` | them back, on any rank, with any number of ranks |
| `write_boxes(file, path, boxes)`, `read_boxes<dim>(file, path)` | a list of boxes the same on every rank (e.g. what a level was clustered to), and back |
| `write_rows(file, path, rows, spans)` | collective: the dataset is created at the total, then each rank writes its patches' rows straight from their memory, one span per local patch |
| `read_rows(file, path, offset, span)` | `span.size()` rows from `offset` into memory, in place |

A span is a `StridedSpan` (a member of an AoS, a column, ...) or a `BoxRegion` (a field's values
on its domain, in C order).

## Layout of a level

`group/boxes` (patches x 2 dim: lower then upper), `group/ids`, `group/ranks`, `group/offsets`.

## No staging copy

A `StridedSpan`'s stride, or a `BoxRegion`'s sub box, is the memory side's hyperslab: HDF5 reads
and writes the user's memory directly.

## Sizes and collectives

- Rows and offsets are 64 bit: a 3D run's particles overflow 32 bits (VTKHDF's offsets are 64
  bit too).
- Writes are independent: each rank writes its own rows.
- Every rank creates every dataset, as HDF5 needs.
