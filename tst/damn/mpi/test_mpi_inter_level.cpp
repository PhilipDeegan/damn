// damn::mpi::refine and coarsen: the serial tests' brute force expectations, the fine patches
// on other ranks than the coarse ones under them (inter_level_tests.hpp). Any number of ranks:
//   mpirun -n 3 ./bin/test_damn_mpi/test_mpi_inter_level

#include "damn/mpi/inter_level.hpp"
#include "damn/mpi/topology.hpp"
#include "damn/mpi/damn_mpi.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/amr/topology.hpp"

#include "gtest/gtest.h"

#include <vector>

namespace damn::amr
{
namespace
{
    int test_rank()
    {
        int r = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &r);
        return r;
    }
    int test_ranks()
    {
        int s = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &s);
        return s;
    }
    int test_ratio()
    {
        return 2;
    }
    Topology test_topology()
    {
        return mpi::topology();
    }

    void do_refine(auto const& op, LevelRegion const region, LevelMap<2>& map,
                   std::vector<LevelPair> const& pairs, auto& coarse, auto& fine,
                   double const alpha)
    {
        mpi::refine(op, region, map, pairs, coarse, fine, alpha);
    }

    void do_refine_by(auto const& access, auto const& op, LevelRegion const region,
                      LevelMap<2>& map, std::vector<LevelPair> const& pairs, auto& coarse,
                      auto& fine, double const alpha)
    {
        mpi::refine(op, region, map, pairs, access, coarse, fine, alpha);
    }

    void do_coarsen_by(auto const& access, auto const& op, LevelMap<2>& map,
                       std::vector<LevelPair> const& pairs, auto& fine, auto& coarse)
    {
        mpi::coarsen(op, map, pairs, access, fine, coarse);
    }

    void do_coarsen(auto const& op, LevelMap<2>& map, std::vector<LevelPair> const& pairs,
                    auto& fine, auto& coarse)
    {
        mpi::coarsen(op, map, pairs, fine, coarse);
    }
} // namespace
} // namespace damn::amr

#include "tst/damn/amr/inter_level_tests.hpp" // IWYU pragma: keep


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    auto const result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
