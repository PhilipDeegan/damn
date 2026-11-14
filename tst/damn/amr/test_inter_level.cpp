// damn::amr::refine and coarsen on one rank, see inter_level_tests.hpp

#include "damn/amr/inter_level.hpp"
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
        return 0;
    }
    int test_ranks()
    {
        return 1;
    }
    Topology test_topology()
    {
        return single_node();
    }

    void do_refine(auto const& op, LevelRegion const region, LevelMap<2>& map,
                   std::vector<LevelPair> const& pairs, auto& coarse, auto& fine,
                   double const alpha)
    {
        refine(op, region, map, pairs, coarse, fine, alpha);
    }

    void do_coarsen(auto const& op, LevelMap<2>& map, std::vector<LevelPair> const& pairs,
                    auto& fine, auto& coarse)
    {
        coarsen(op, map, pairs, fine, coarse);
    }
} // namespace
} // namespace damn::amr

#include "tst/damn/amr/inter_level_tests.hpp" // IWYU pragma: keep


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
