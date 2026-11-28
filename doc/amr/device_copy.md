# Copies on the device

GPU builds copy resources in unified memory on the device. The host never touches their values:
not to copy them, not to pack them for another rank.

## Which resources qualify

A resource is on the device when its type says so (`on_device`). The device copy takes it
(`DeviceBoxSpan`) when it also gives `box_span(resource, patch box)`: its data and the box of
its points, in C order. The kernels further need `double` values: a source read through a
pointer to (const) `double`, a destination written through a pointer to `double`.

Every side of every piece of a plan (local pieces, pieces from and to other ranks) is classed:

| Side | Then |
|---|---|
| on the device, taken | the device path |
| not on the device | the host path |
| on the device, not taken | an error |

A plan whose sides are all the device's runs on the device, one whose sides are all the host's
runs on the host. A plan mixing them, or with a device resource the device copy cannot take,
throws: the host path would read device values on the host.

## Operations

Leaving items (Append) have their own device path: `doc/amr/device_items.md`.


| Function | Copy | How |
|---|---|---|
| `copy_on_device<arrival>(plan, access, ld)` | Assign, Sum, Max on one rank | `finish_on_device`; false for host resources |
| `pack_on_device(table)` | messages | the outgoing pieces into the table's send buffer, one launch |
| `finish_on_device<arrival>(plan, table, ld)` | Assign, Sum, Max | Assign: every piece's points in one launch, a thread per point, `dst(p) = src(p - shift)`; Sum and Max: every piece's source values snapshot into a device buffer (one launch), then a thread per destination point folds the contributions |

A thread finds its piece by binary search over the pieces' first points. Pieces from another
rank read their values from the table's receive buffer, so `finish_on_device` runs once the
messages are in (`damn::mpi::copy`, `doc/mpi/copy.md`).

### Fold order

The sum starts from 0 and adds the destination's own value at its place among the pieces'
images, in the same order as the host path's `fold_sums`: pieces grouped by destination and
resources, then ordered by source image. The result has the host path's bits. The max starts
from the destination's own value and takes each contribution as `std::max` does (the value so
far, unless less than the next): the host path's result.

## The device table

The table is built on the plan's first copy and kept with it: the pieces, the outgoing pieces,
and per peer rank its values in the send and receive buffers. A message holds its pieces' values
in plan order, peers in rank order: the host path's bytes, so a rank on either path reads it.
Plans live with their level's neighbours, so a new decomposition makes new plans and so new
tables. The fold groups are built on the first sum or max.

A plan keeps its first copy's table: a later copy of the same plan (region and resource names)
through accessors reaching resources of the other side uses it all the same.
