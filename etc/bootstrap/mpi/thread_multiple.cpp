// bootstrap probe (see etc/bootstrap/bootstrap.py): ranks=2 profile=bootstrap
// MPI_THREAD_MULTIPLE provided, and two threads per rank exchanging with the other rank at the
// same time, on one communicator (distinct tags) and on a duplicate each, complete and agree.
// HDF5's subfiling needs it; Open MPI over UCX needs UCX built --enable-mt for it
#include <mpi.h>

#include <array>
#include <cstdio>
#include <thread>

int main(int argc, char** argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    std::printf("provided=%d\nmultiple=%d\n", provided, MPI_THREAD_MULTIPLE);
    if (provided < MPI_THREAD_MULTIPLE)
    {
        MPI_Finalize();
        return 1;
    }

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int const other = (rank + 1) % size;
    std::array<MPI_Comm, 2> comms{};
    for (auto& c : comms)
        MPI_Comm_dup(MPI_COMM_WORLD, &c);

    int constexpr n = 1 << 16, rounds = 64;
    std::array<int, 4> ok{1, 1, 1, 1};
    auto const exchange = [&](int const t, MPI_Comm const comm, int const tag) {
        static thread_local std::array<int, n> out, in;
        for (int r = 0; r < rounds; ++r)
        {
            for (int i = 0; i < n; ++i)
                out[i] = rank * 1000003 + t * 7919 + r * 31 + i;
            MPI_Request reqs[2];
            MPI_Irecv(in.data(), n, MPI_INT, (rank + size - 1) % size, tag, comm, &reqs[0]);
            MPI_Isend(out.data(), n, MPI_INT, other, tag, comm, &reqs[1]);
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
            int const from = (rank + size - 1) % size;
            for (int i = 0; i < n; ++i)
                ok[t] &= in[i] == from * 1000003 + t * 7919 + r * 31 + i;
        }
    };
    {
        std::array<std::thread, 4> threads{
            std::thread{exchange, 0, MPI_COMM_WORLD, 10}, // one communicator, two tags
            std::thread{exchange, 1, MPI_COMM_WORLD, 11},
            std::thread{exchange, 2, comms[0], 0}, // a communicator each
            std::thread{exchange, 3, comms[1], 0}};
        for (auto& t : threads)
            t.join();
    }
    int mine = ok[0] & ok[1] & ok[2] & ok[3], all = 0;
    MPI_Allreduce(&mine, &all, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    std::printf("exchanges=%s\n", all ? "ok" : "corrupt");
    for (auto& c : comms)
        MPI_Comm_free(&c);
    MPI_Finalize();
    return all ? 0 : 1;
}
