# Transfers between levels

`inc/damn/amr/inter_level.hpp` moves data between two consecutive levels, along a `LevelMap`.
damn finds what goes where and moves it; the user's operators do the arithmetic, since they know
the centerings and the physics.

Sources are never modified. An operator handles the resource kinds it has a call for: one asked
to move a resource it has no call for throws.

## Refine

```cpp
refine(op, region, map, pairs, coarse_ld, fine_ld, alpha = 0);
```

### Regions

| Region | Fine targets |
|---|---|
| `Interior` | the fine patches' domain points (or cells), e.g. for a new level |
| `LevelGhosts` | fine ghost points (or cells) on no fine patch's domain, periodic images included: the coarse-fine border |

### Gathering the coarse values

For each fine target box, the coarse points (or cells) of `grow(coarsen(box), op.stencil)` are
gathered from the coarse patches' domains.

Field points on no coarse domain lie past the coarse level's border. This happens when a fine
level is nearer the coarse level's border than its ghosts' reach (with `nesting_buffer` 1, for
example). They are taken from the first coarse patch, in level order, whose ghosts hold them:
the coarse level's level ghosts, filled from the level below it. So a level only ever reads the
next coarser one. The choice is made from the coarse level's geometry, so it is the same on every
rank.

Points with no coarse data at all (past a non periodic edge) stay NaN (or value initialized,
for types without a NaN): the gathered values are cleared before every refine, so they are never
stale.

### Operators

| Data | Call | Does |
|---|---|---|
| `BoxData` | `op(fine_resource, fine_patch_box, box, BoxValues<T const> coarse)` | fills the fine points of `box` from the coarse values |
| `CellItemData` | `op(fine_resource, fine_patch_box, box, std::span<Item const> coarse, shift)` | per coarse patch image, appends the fine items (e.g. split coarse ones) whose cell is in `box`; a coarse item's cell is at `cell(item) + shift` in the coarse index space |

### Time interpolation

With a pair's `coarse_new` set, the coarse values are `(1 - alpha) coarse + alpha coarse_new`.
At `alpha` 0 only `coarse` is read, at `alpha` 1 only `coarse_new`, so the other may be unset
(`0 * NaN` is NaN, so blending would not do).

## Coarsen

```cpp
coarsen(op, map, pairs, fine_ld, coarse_ld);
```

`BoxData` only. For each fine patch and each coarse patch image under it,
`op(fine_resource const&, fine_patch_box, BoxValues<T> out)` fills `out.box`: the coarse domain
points under the fine patch's domain. The fine ghosts are expected filled. The values are then
assigned to the coarse patch.

## Resources

How the transfers reach a pair's resources on a patch's data follows `damn::amr::copy`
([copy.md](copy.md)):

- **`by_name`** (the default);
- **accessors**: the caller's, one per pair in the pairs' order, with `access[i].coarse(data)`,
  `.coarse_new(data)` (for pairs interpolated in time) and `.fine(data)` returning the typed
  resources (a reference, or a handle holding a pointer).

Plans are still made from the names. Resources' boxes come from the user's `domain_box` /
`ghost_box` for the resource kind, from any patch data holding the name on either level: the
coarse and fine resources of a name are expected to be of the same kind. A rank without patches
of one level still plans the other level's sends, taking that level's resources from its
prototype.

## Plans and ranks

What goes where is planned on first use for a region, a set of resources and a stencil, and kept
in the `LevelMap` ([level_map.md](level_map.md)); the map must be rebuilt after a regrid. A plan holds targets (a fine patch's
box and the coarse box it needs) and pieces (the part of a target coming from one coarse patch
image).

The functions in `inter_level.hpp` work on one rank and throw for pieces of other ranks; those
go through `damn/mpi/inter_level.hpp`. Both use the same packing functions:

| Function | Moves |
|---|---|
| `pack_coarse_piece` / `unpack_coarse_piece` | refine: a piece's coarse data (values in the box's point order, or the item count then the items) into the fine target's scratch, or the items to `op` |
| `pack_coarsened` / `unpack_coarsened` | coarsen: a target's coarse values computed by `op` into the coarse patch |
