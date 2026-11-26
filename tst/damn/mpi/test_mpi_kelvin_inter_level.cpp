// the fine patches on other ranks than the coarse ones under them; any number of ranks

#include "damn/mpi/inter_level.hpp"
#include "damn/mpi/mpi_utils.hpp"
#include "damn/mpi/topology.hpp"
#include "damn/mpi/damn_mpi.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/amr/topology.hpp"

#include "gtest/gtest.h"

#include <vector>

namespace damn::amr::kelvin
{
namespace
{
    int test_rank()
    {
        return mpi::rank();
    }
    int test_ranks()
    {
        return mpi::size();
    }
    Topology test_topology()
    {
        return mpi::topology();
    }
    void do_refine(auto const& op, LevelRegion const region, LevelMap<3>& map,
                   std::vector<LevelPair> const& pairs, auto& coarse, auto& fine,
                   double const alpha)
    {
        mpi::refine(op, region, map, pairs, coarse, fine, alpha);
    }
    void do_coarsen(auto const& op, LevelMap<3>& map, std::vector<LevelPair> const& pairs,
                    auto& fine, auto& coarse)
    {
        mpi::coarsen(op, map, pairs, fine, coarse);
    }
} // namespace
} // namespace damn::amr::kelvin

#include "tst/damn/amr/kelvin_inter_level_tests.hpp" // IWYU pragma: keep


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    auto const result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
