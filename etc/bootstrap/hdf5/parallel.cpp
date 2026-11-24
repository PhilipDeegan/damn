// bootstrap probe (see etc/bootstrap/bootstrap.py): ranks=2 profile=bootstrap_hdf5 needs=mpi_found,hdf5_parallel
// parallel HDF5 through MPI-IO: each rank writes its rows of one dataset, collectively, and reads
// every rank's back. Fails to build without H5_HAVE_PARALLEL
#include <hdf5.h>
#include <mpi.h>
#if !H5_HAVE_PARALLEL
#error "HDF5 not parallel"
#endif

#include <cstdio>
#include <vector>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::printf("version=%d.%d.%d\n", H5_VERS_MAJOR, H5_VERS_MINOR, H5_VERS_RELEASE);

    hsize_t constexpr rows = 1 << 16;
    hsize_t const total[1] = {rows * static_cast<hsize_t>(size)}, mine[1] = {rows},
                  first[1] = {rows * static_cast<hsize_t>(rank)};
    std::vector<double> out(rows), in(total[0]);
    for (hsize_t i = 0; i < rows; ++i)
        out[i] = static_cast<double>(first[0] + i);

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
    hid_t file  = H5Fcreate("parallel.h5", H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    hid_t space = H5Screate_simple(1, total, nullptr);
    hid_t ds
        = H5Dcreate2(file, "x", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t mem = H5Screate_simple(1, mine, nullptr);
    H5Sselect_hyperslab(space, H5S_SELECT_SET, first, nullptr, mine, nullptr);
    hid_t xfer = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer, H5FD_MPIO_COLLECTIVE);
    herr_t const w = H5Dwrite(ds, H5T_NATIVE_DOUBLE, mem, space, xfer, out.data());
    H5Sselect_all(space);
    herr_t const r = H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, xfer, in.data());
    H5Pclose(xfer);
    H5Sclose(mem);
    H5Dclose(ds);
    H5Sclose(space);
    H5Fclose(file);
    H5Pclose(fapl);

    int good = w >= 0 and r >= 0;
    for (hsize_t i = 0; good and i < total[0]; ++i)
        good = in[i] == static_cast<double>(i);
    int all = 0;
    MPI_Allreduce(&good, &all, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    std::printf("roundtrip=%s\n", all ? "ok" : "failed");
    MPI_Finalize();
    return all ? 0 : 1;
}
