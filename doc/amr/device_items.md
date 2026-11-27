# Leaving items on the device

GPU builds with thrust move a `gather_leaving` copy's items in unified memory on the device
(`inc/damn/amr/device_items.hpp`), the host never touching them. damn sees an item resource as
columns, whatever the items are.

## Hooks

A resource on the device (`on_device`) gives, found by ADL:

```cpp
DeviceItems<dim>    device_items(src const&);              // its items, grouped by cell
DeviceItemsOut<dim> device_items_appended(dst&, std::size_t n); // dst grown by n items
```

| Type | Holds |
|---|---|
| `DeviceItems` | `cells`, and over them in C order per cell its first item (`starts`) and number (`sizes`): a cell's items are a run in every column; the columns, and their `layout` (an item's bytes per column) |
| `DeviceItemsOut` | the new items' columns (from the first new item), their `layout`, and per direction an `int` column for their cell |

The cell columns are left out of `layout`: an item's cell is its run's cell, so messages do not
carry them and the destination's are written from the run. `device_items` throws if its items
are not grouped by cell (e.g. not sorted since appended to). After the copy
`cell_items::appended(dst)` runs once per destination, e.g. to group them again.

A plan's sides are classed as for fields (`doc/amr/device_copy.md`): every resource on the device
and taken, the device path; none on the device, the host path; else an error.

## Blocks

A piece's items go through a block: its cells' item offsets (CSR over the piece's destination
cells, C order, cells + 1 `uint32`), then each column, 8 byte aligned. The local pieces' blocks
come first in the plan's send buffer, then per peer rank (rank order) its pieces' blocks in plan
order.

| Step | What |
|---|---|
| counts | a thread per (piece, cell): its source cell's items |
| scan | an exclusive scan of the counts (thrust); per piece its items, read back (one value per piece) |
| offsets | a thread per (piece, cell): the block's CSR |
| pack | a thread per item: its cell by binary search over the scan, its columns into the block |
| unpack | per destination one `device_items_appended` with all its items; a thread per item: its cell by binary search over its block's offsets, its columns and cell into the destination |

Local pieces are packed too, so a source may be its own destination: it grows only once
every source is read.

## Over MPI

Per peer two messages: a counts message (host), per piece its items and layout (3 words, its
size the plan's), then the peer's blocks (device). A receiver posts the counts' receives first,
then the blocks' at the place the counts give, and checks each message's size against them.
