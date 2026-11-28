#ifndef DAMN_TST_AMR_TEST_INTER_LEVEL_HPP
#define DAMN_TST_AMR_TEST_INTER_LEVEL_HPP

// one rank, at the ratio DAMN_TEST_INTER_LEVEL_RATIO the including file defines

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
    int test_ratio()
    {
        return DAMN_TEST_INTER_LEVEL_RATIO;
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

    void do_refine_by(auto const& access, auto const& op, LevelRegion const region,
                      LevelMap<2>& map, std::vector<LevelPair> const& pairs, auto& coarse,
                      auto& fine, double const alpha)
    {
        refine(op, region, map, pairs, access, coarse, fine, alpha);
    }

    void do_coarsen_by(auto const& access, auto const& op, LevelMap<2>& map,
                       std::vector<LevelPair> const& pairs, auto& fine, auto& coarse)
    {
        coarsen(op, map, pairs, access, fine, coarse);
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

#endif // DAMN_TST_AMR_TEST_INTER_LEVEL_HPP
