# HDF5 files (`hdf5/h5_file.hpp`)

`HighFiveFile` is an HDF5 file through HighFive. It opens serially or in parallel (MPI-IO over `MPI_COMM_WORLD`),
and reads and writes datasets (flat, nested vectors, chunked, resizable) and attributes.

## Every rank creates everything

HDF5 needs every rank to create every dataset and attribute, collectively. The `*_per_mpi`
functions gather paths, sizes and values with `damn::mpi`, so each rank creates all of them,
then writes only its own. A rank with nothing to write passes a size of 0 (datasets) or an empty
path (attributes) and still takes part. Similar sized paths on every rank are best.

Missing parent groups are created on the way: HighFive does not.

## Subfiling

HDF5's subfiling VFD can write parallel files as several subfiles.

| Setting | Effect |
|---|---|
| `DAMN_H5_SUB_FILES=<n>` | `n` subfiles per file, `n` the stripe count; 0 or unset: MPI-IO |
| `H5FD_SUBFILING_STRIPE_SIZE`, `H5FD_SUBFILING_IOC_PER_NODE`, `H5FD_IOC_THREAD_POOL_SIZE` | HDF5's own; otherwise its defaults (one I/O concentrator per node) |

- The I/O concentrators are threads calling MPI, so MPI must run at `MPI_THREAD_MULTIPLE`:
  `mpi_thread_multiple_required()` tells whoever starts MPI.
- Asked for and not possible (HDF5 without the VFD, MPI below `MPI_THREAD_MULTIPLE`): opening the
  file throws, saying why.
- Subfiled files are read through the same driver, or merged first with HDF5's `h5fuse`.
