// bootstrap probe (see etc/bootstrap/bootstrap.py): ranks=1 profile=bootstrap
// Open MPI's CUDA awareness: built with it (MPIX_CUDA_AWARE_SUPPORT) and available at run time
// (MPIX_Query_cuda_support: a CUDA driver found). Fails to build without Open MPI's extension
#include <mpi.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#if !defined(MPIX_CUDA_AWARE_SUPPORT) || !MPIX_CUDA_AWARE_SUPPORT
#error "MPI not built CUDA aware"
#endif

#include <cstdio>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int const runtime = MPIX_Query_cuda_support();
    std::printf("compiled=1\nruntime=%d\n", runtime);
    MPI_Finalize();
    return runtime ? 0 : 1;
}
