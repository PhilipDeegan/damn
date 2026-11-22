// the exchange probes' code (exchange_<distance>.cpp): two ranks, placed by the probe's mpirun
// args, exchange messages both ways at once (Irecv, Isend, Waitall, as the border copies) over a
// range of sizes; the best of several repeats per size, fitted (least squares) to
// latency + bytes * seconds_per_byte (relative error). Prints latency_s, s_per_byte,
// s_per_byte_large (the two largest sizes: past the cache), bytes_max and where the ranks
// were (cpu, numa node, same shared memory node); exits 1 when the placement is not the
// distance's (same_numa must be `numa`)
#include <mpi.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

inline int exchange_probe(int argc, char** argv, bool const same_numa)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2)
    {
        std::printf("error=2 ranks needed\n");
        MPI_Finalize();
        return 1;
    }

    // where this rank runs: its cpu and NUMA node, and whether both share memory
    unsigned cpu = 0, numa = 0;
    getcpu(&cpu, &numa);
    int numas[2]{};
    int const mine = static_cast<int>(numa);
    MPI_Allgather(&mine, 1, MPI_INT, numas, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Comm shared;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    int shared_size = 0;
    MPI_Comm_size(shared, &shared_size);
    MPI_Comm_free(&shared);
    bool const same_node = shared_size == 2, numa_same = numas[0] == numas[1];
    std::printf("numa0=%d\nnuma1=%d\nsame_node=%d\n", numas[0], numas[1], int{same_node});
    if (!same_node or numa_same != same_numa)
    {
        std::printf("error=placement is not this distance's\n");
        MPI_Finalize();
        return 1;
    }

    int const other = 1 - rank, repeats = 5;
    std::vector<double> bytes, seconds;
    // up to 128 MB: past the last level cache (this box's: 96 MB), so the largest are memory's
    for (std::size_t n = 8; n <= (std::size_t{128} << 20); n *= 4)
    {
        std::vector<char> out(n, static_cast<char>(rank)), in(n);
        int const rounds
            = static_cast<int>(std::clamp<std::size_t>((std::size_t{256} << 20) / n, 4, 4096));
        double best = 1e300;
        for (int k = 0; k < repeats; ++k)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            auto const t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < rounds; ++r)
            {
                MPI_Request reqs[2];
                MPI_Irecv(in.data(), static_cast<int>(n), MPI_CHAR, other, 0, MPI_COMM_WORLD,
                          &reqs[0]);
                MPI_Isend(out.data(), static_cast<int>(n), MPI_CHAR, other, 0, MPI_COMM_WORLD,
                          &reqs[1]);
                MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
            }
            double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                       / rounds;
            MPI_Allreduce(MPI_IN_PLACE, &t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            best = std::min(best, t);
        }
        bytes.push_back(static_cast<double>(n));
        seconds.push_back(best);
    }

    // least squares in relative error (weights 1 / seconds^2: small and large messages count
    //  alike): seconds = a + bytes * per_byte; latency the smallest message's time; and the
    //  slope between the two largest
    //  (past the cache: memory's)
    double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i)
    {
        double const w = 1. / (seconds[i] * seconds[i]);
        sw += w;
        sx += w * bytes[i];
        sy += w * seconds[i];
        sxx += w * bytes[i] * bytes[i];
        sxy += w * bytes[i] * seconds[i];
    }
    double const per_byte = (sw * sxy - sx * sy) / (sw * sxx - sx * sx);
    double const latency  = seconds[0]; // the smallest message's: its bytes cost nothing
    auto const k          = bytes.size() - 1;
    double const large    = (seconds[k] - seconds[k - 1]) / (bytes[k] - bytes[k - 1]);
    if (rank == 0)
        std::printf("latency_s=%.6g\ns_per_byte=%.6g\ns_per_byte_large=%.6g\nbytes_max=%.0f\n",
                    latency, per_byte, large, bytes.back());
    MPI_Finalize();
    return per_byte > 0 ? 0 : 1;
}
