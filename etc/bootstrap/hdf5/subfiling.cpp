// bootstrap probe (see etc/bootstrap/bootstrap.py): ranks=2 profile=bootstrap_hdf5
// needs=mpi_found,hdf5_subfiling
// HDF5's subfiling VFD: built with it (H5_HAVE_SUBFILING_VFD), MPI at MPI_THREAD_MULTIPLE (its I/O
// concentrator threads call MPI), and a file written by every rank through it read back. Fails
// to build without the VFD
#include <hdf5.h>
#include <mpi.h>
#if !H5_HAVE_SUBFILING_VFD
#error "HDF5 without the subfiling VFD"
#endif

#include <cstdio>
#include <vector>

int main(int argc, char** argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::printf("version=%d.%d.%d\nthread_multiple=%d\n", H5_VERS_MAJOR, H5_VERS_MINOR,
                H5_VERS_RELEASE, provided >= MPI_THREAD_MULTIPLE);
    if (provided < MPI_THREAD_MULTIPLE)
    {
        std::printf("reason=MPI_THREAD_MULTIPLE not provided\n");
        MPI_Finalize();
        return 1;
    }

    hsize_t constexpr rows = 1 << 16;
    hsize_t const total[1] = {rows * static_cast<hsize_t>(size)}, mine[1] = {rows},
                  first[1] = {rows * static_cast<hsize_t>(rank)};
    std::vector<double> out(rows), in(total[0]);
    for (hsize_t i = 0; i < rows; ++i)
        out[i] = static_cast<double>(first[0] + i);

    auto const fapl_of = [] {
        hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_mpi_params(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
        H5Pset_fapl_subfiling(fapl, nullptr); // defaults: an I/O concentrator per node
        return fapl;
    };
    int good = 1;
    {
        hid_t fapl  = fapl_of();
        hid_t file  = H5Fcreate("subfiling.h5", H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        hid_t space = H5Screate_simple(1, total, nullptr);
        hid_t ds    = file < 0 ? -1
                               : H5Dcreate2(file, "x", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT,
                                            H5P_DEFAULT, H5P_DEFAULT);
        hid_t mem   = H5Screate_simple(1, mine, nullptr);
        H5Sselect_hyperslab(space, H5S_SELECT_SET, first, nullptr, mine, nullptr);
        good = file >= 0 and ds >= 0
               and H5Dwrite(ds, H5T_NATIVE_DOUBLE, mem, space, H5P_DEFAULT, out.data()) >= 0;
        H5Sclose(mem);
        H5Sclose(space);
        if (ds >= 0)
            H5Dclose(ds);
        if (file >= 0)
            H5Fclose(file);
        H5Pclose(fapl);
    }
    if (good)
    {
        hid_t fapl = fapl_of();
        hid_t file = H5Fopen("subfiling.h5", H5F_ACC_RDONLY, fapl);
        hid_t ds   = file < 0 ? -1 : H5Dopen2(file, "x", H5P_DEFAULT);
        good = ds >= 0
               and H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, in.data()) >= 0;
        for (hsize_t i = 0; good and i < total[0]; ++i)
            good = in[i] == static_cast<double>(i);
        if (ds >= 0)
            H5Dclose(ds);
        if (file >= 0)
            H5Fclose(file);
        H5Pclose(fapl);
    }
    int all = 0;
    MPI_Allreduce(&good, &all, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    std::printf("roundtrip=%s\n", all ? "ok" : "failed");
    MPI_Finalize();
    return all ? 0 : 1;
}
