#ifndef DAMN_MPI_DAMN_MPI_HPP
#define DAMN_MPI_DAMN_MPI_HPP

// include mpi.h through this file only: it silences mpi.h's own warnings

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
