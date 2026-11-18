#ifndef DAMN_MPI_DAMN_MPI_HPP
#define DAMN_MPI_DAMN_MPI_HPP

// include mpi.h through this file only: it silences mpi.h's own warnings, and skips the MPI C++
//  bindings (MPICH's mpicxx.h, Open MPI 4's ompi/mpi/cxx): unused, deprecated, and they drag
//  in their own headers and link libraries (res/mkn/mpi.yaml defines these for every file, as
//  parallel HDF5's hdf5.h includes mpi.h itself)
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX 1
#endif
#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX 1
#endif

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wbad-function-cast"
#endif
#endif

#if __has_include("mpi.h") // silence clangd
#include "mpi.h" // IWYU pragma: export
#endif

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#endif /* DAMN_MPI_DAMN_MPI_HPP */
