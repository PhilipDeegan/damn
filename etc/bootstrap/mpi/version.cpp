// bootstrap probe (see etc/bootstrap/bootstrap.py): ranks=1 profile=bootstrap needs=mpi_found
// the MPI library and the standard it implements; always passes when MPI builds and runs
#include <mpi.h>

#include <cstdio>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    char version[MPI_MAX_LIBRARY_VERSION_STRING];
    int len = 0, major = 0, minor = 0;
    MPI_Get_library_version(version, &len);
    MPI_Get_version(&major, &minor);
    for (int i = 0; i < len; ++i) // one line
        if (version[i] == '\n')
            version[i] = ' ';
    std::printf("library=%.*s\nstandard=%d.%d\n", len, version, major, minor);
    MPI_Finalize();
}
