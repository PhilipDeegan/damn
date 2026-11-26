# Moving cell items

How a `CellItemData` resource's items move between patches: the items whose `cell + shift` is in
a box are either packed to bytes and unpacked (other ranks, regrid transfers), or appended
directly (the same rank).

A resource can do each step its own way through optional free functions, found by ADL. Without
them, damn does it item by item with the `CellItemData` functions (`for_each_item`, `cell`,
`append`) and the item's bytes.

## Hooks

| Hook | Does |
|---|---|
| `void pack_items(src const&, box, shift, std::vector<std::byte>&)` | appends the selected items in the resource's own format, e.g. a count then one byte stream per component |
| `void unpack_items(dst&, std::byte const*& at, shift)` | appends what `pack_items` wrote, cells moved by `shift`; `at` moves past it |
| `std::vector<value_type> read_items(dst const&, std::byte const*& at)` | what `pack_items` wrote, as items; `at` moves past it (refine's coarse items) |
| `void append_items_from(dst&, src const&, box, shift)` | the same rank, directly (`src` may be `dst`) |
| `void items_appended(dst&)` | called once after a copy or a regrid appended to `dst`, whatever the number of pieces, e.g. to sort it |

`pack_items` and `unpack_items` come together. Without them the format is a count
(`std::uint64_t`), then the items' bytes.

Without `append_items_from`, a resource with `pack_items` goes through a scratch buffer. The
buffer also makes appending to `src` itself safe, as does collecting first in the item by item
path: appending to the same array would invalidate the iteration.
