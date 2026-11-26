# Copies within a level

`inc/damn/amr/copy.hpp` copies data between the patches of one level, along the links of the
level's `Neighbours`. Sources are never modified: they are neither cleared nor overwritten.

## Operations

A copy is what goes where (a `Region`) and what happens on arrival (an `Arrival`):

| Operation | Region | Arrival | Data | Effect |
|---|---|---|---|---|
| `fill_ghosts` | `Ghosts` | `Assign` | `BoxData` | neighbours' domain values into my ghosts; my own domain points are never written |
| `sum_borders` | `Borders` | `Sum` | `BoxData` | where ghost boxes overlap, every patch ends with the sum of all patches' values (deposits), the same bits on every patch (see [Deterministic sums](#deterministic-sums)) |
| `max_borders` | `Borders` | `Max` | `BoxData` | as `sum_borders`, with the max |
| `gather_leaving` | `Leaving` | `Append` | `CellItemData` | items a neighbour holds outside its domain are appended to the domain of the patch owning their cell; the source arrays are not cleared |

```cpp
copy(fill_ghosts, neighbours, {"Bx", "By"}, level_data);
```

## Resources

Resources come as (source, destination) pairs (`ResourcePair`), or as names for fields, where
both sides have the same name.

How a copy reaches a piece's resources on a patch's data:

- **`by_name`** (the default): looked up by their names (`for_each_resource`).
- **accessors**: the caller's, one per pair in the pairs' order (a `std::vector`, for example).
  `access[piece.pair].src(patch_data)` and `.dst(patch_data)` return the typed resources
  directly (a reference, or a handle holding a pointer), with no lookup. The names are still
  used to plan: they give the geometry and the plan's key.

## Plans

What goes where is planned on first use for a region and a set of resources, and kept in the
`Neighbours`; the neighbours must be rebuilt after a regrid. A plan is a list of pieces: a box,
a source link and a destination patch.

- `Ghosts`: a primal node on a patch border is a domain point of every patch sharing it (and of
  its periodic image). The first link covering a ghost point fills it, so exactly one piece
  writes it, and every rank plans the same one because links are in level order.
- `Borders`: every overlap of ghost boxes is a piece. Local pieces get an offset in the plan's
  buffer.
- `Leaving`: the destination's domain against each neighbour's ghost box.

Pieces going to another rank's patch are planned too (the plan's `outgoing`), from the links of
the remote destinations.

## How a piece goes

This follows from its link's `Distance`:

| Distance | How |
|---|---|
| `Self`, `Local` | directly, patch to patch; sums and maxes go through one buffer kept in the plan, so they use the values from before the copy |
| `Numa` | as `Local`, for now |
| `Node`, `Remote` | `damn::amr::copy` throws; `damn::mpi::copy` (`damn/mpi/copy.hpp`) sends them, with the pieces packed by `pack_piece` and unpacked by `unpack_piece` |

GPU builds do the copy on the device (`device_copy.hpp`) when every piece's resources are on the
device, for `Assign`, `Sum` and `Append`; otherwise the host path does it.

A `BoxData` whose values are a C ordered array over a box (`SpanBoxData`: `box_span` gives its
data and the box of its points) is copied span by span, a run along the last direction being
contiguous in memory, instead of point by point. The point order is the same either way, so the
same bytes arrive and the same sums come out.

## Deterministic sums

A point on a border is held by several patches, and each must end with the same bits. Floating
point addition is not associative (`(a + b) + c` and `(a + c) + b` can differ), so the order of
the contributions must be the same on every holder.

`fold_sums`, per destination and resource:

1. the summed points are zeroed (the destination's own values kept aside);
2. the contributions are added in the lexicographic order of their patch's box in the
   destination's space (its image), the destination's own included at its place in that order.

Every holder of a point has the same contributing patches, their images one shift apart, and the
lexicographic order is translation invariant: the same values are added in the same order, giving
the same bits.

`finish_plan` sorts the pieces into this order once per plan (`fold_order`) and groups them by
destination and resource (`fold_groups`), each group with the points it covers, each point once.
