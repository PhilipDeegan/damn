# Checkpoints (`hdf5/checkpoint.hpp`)

A level's resources, by name, in one HDF5 group, written and read through level tables
(`hdf5/level_table.hpp`, `doc/hdf5/level_table.md`).

| Function | What |
|---|---|
| `write_checkpoint(file, group, level, data, names)` | collective |
| `read_checkpoint(file, group, level, data, names)` | into `data`, on the same patches as written: the same ranks and decomposition. BoxData resources are read on their checkpoint box, columns resized to their rows first |

Each also takes a `LevelCheckpoint{file, group, level, data, names}`.

## Layout

`group/level` holds the level's boxes, ids and owners. Each resource has `group/<name>/offsets`
and its datasets, by kind:

| Kind | Datasets | Resource provides (ADL) |
|---|---|---|
| BoxData (fields) | `group/<name>/values`: each patch's values on its checkpoint box, in C order (the last direction fastest) | optionally `checkpoint_box(resource, patch_box)`, else the domain box: e.g. the ghost box, for ghosts that cannot be rebuilt. Written and read in place with `box_span(resource, patch_box)` (a `core::BoxSpan` over its ghost box, a C ordered buffer), else through `value()` per point |
| columns | `group/<name>/<column>` | `for_each_column(resource, fn(name, mkn::kul::StridedSpan<T>))` and `resize_columns(resource, n)`, e.g. a particle array's components: each one a column, a row per item. Written and read in place through the spans |

## Collectives

damn knows nothing of what the resources are: the user's ADL functions say. Every rank calls
every collective, with patches or not. A rank without patches learns each resource's kind and
columns from the others: the schema, a string per resource, is gathered from the ranks that
have it (`"b<type>"` for BoxData values, or `"c"` then `"<type><name>;"` per column).
