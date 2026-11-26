# Copies on the device

GPU builds run a copy's same rank pieces on the device, for resources in unified memory whose
values the host never touches.

## Which resources qualify

A resource qualifies (`DeviceBoxSpan`) when:

- its type says it is on the device (`on_device`), and
- it gives `box_span(resource, patch box)`: its data and the box of its points, in C order.

The assign and sum kernels further need `double` values: a source read through a pointer to
(const) `double`, a destination written through a pointer to `double`.

Every local piece of the plan must qualify; otherwise the device functions return false, do
nothing, and the copy's host path runs.

## Operations

| Function | Copy | How |
|---|---|---|
| `assign_on_device(plan, access, ld)` | Assign | every piece's points in one launch, a thread per point; the thread's piece is found by binary search over the pieces' first points; `dst(p) = src(p - shift)` |
| `sum_on_device(plan, access, ld)` | Sum | every piece's source values are snapshot into a device buffer (one launch); then a thread per summed destination point adds the contributions in `fold_sums`' order |
| `append_on_device(plan, access, ld)` | Append (items) | per resource pair, every piece handed at once to the resource type's device hooks |

### Sum order

The sum starts from 0 and adds the destination's own value at its place among the pieces'
images, in the same order as the host path's `fold_sums`: pieces grouped by destination and
resources, then ordered by source image. The result has the host path's bits.

### Append hooks

The resource type provides two hooks, found by ADL:

```cpp
bool can_append_items_on_device(std::vector<AppendPiece<Dst, Src, dim>> const& pieces);
void append_items_on_device(std::vector<AppendPiece<Dst, Src, dim>> const& pieces);
```

`pieces` are in plan order, so a destination's items come in the host path's order. An
`AppendPiece` carries the resources (copies: handles), the piece's destination box, its source's
shift, and its destination patch (`LevelData` index).

Nothing is done unless every pair can: a non local piece, a pair without hooks, or a pair whose
`can_append_items_on_device` refuses sends the whole copy to the host path.

## The device table

The pieces' table is built on the plan's first device copy and kept with it. Plans live with
their level's neighbours, so a new decomposition makes new plans and so new tables. The sum
groups are built on the first sum.
