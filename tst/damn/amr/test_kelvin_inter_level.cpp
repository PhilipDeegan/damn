#include "damn/amr/hierarchy.hpp"
#include "damn/amr/inter_level.hpp"
#include "damn/amr/kelvin/inter_level.hpp"
#include "damn/amr/kelvin/site_data.hpp"
#include "damn/amr/kelvin/sites.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/amr/topology.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace damn::amr::kelvin
{
TEST(KelvinInterLevel, eachSublatticeNestsInItself)
{
    for (int const r : {1, 3, 5})
        for (auto const site : {Site::corner, Site::body})
            for (int c = -4; c <= 4; ++c)
            {
                core::Point<int, 1> const coarse{std::array{c}};
                auto const f = coincident_fine(coarse, site, r);
                EXPECT_EQ(f[0], r * c + (site == Site::body ? (r - 1) / 2 : 0));
                EXPECT_EQ(coincident_coarse(f, site, r), coarse);
                for (int i = 0; i < r; ++i) // the fine cells over coarse c: sites at c + num / r
                {
                    core::Point<int, 1> const fine{std::array{r * c + i}};
                    auto const b = bracket(fine, site, r);
                    EXPECT_GE(b.num[0], 0);
                    EXPECT_LT(b.num[0], r);
                    EXPECT_EQ(r * b.lower[0] + b.num[0] + nest_offset(site, r), fine[0]);
                    EXPECT_LE(std::abs(b.lower[0] - c), 1); // stencil 1
                    EXPECT_EQ(coincident_coarse(fine, site, r).has_value(), b.num[0] == 0);
                }
            }
    // positions, in coarse h units: a key K is at K; a fine key F at F / r
    for (int const r : {3, 5})
        for (auto const site : {Site::corner, Site::body})
            for (int f = -7; f <= 7; ++f)
            {
                core::Point<int, 1> const fine{std::array{f}};
                auto const b      = bracket(fine, site, r);
                double const at   = key_of(fine, site)[0] / static_cast<double>(r);
                double const from = key_of(b.lower, site)[0];
                EXPECT_DOUBLE_EQ(at, from + 2. * b.num[0] / r)
                    << r << " " << f; // 2: a cell, h units
            }
    EXPECT_THROW(check_ratio(2), std::runtime_error);
}

// a domain's sites on the finer level: a coarse site's coincident fine site is in it if the coarse
//  site is; periodic, its cells are the coarse ones refined
TEST(KelvinInterLevel, refinedKeys)
{
    KeyBox<3> const periodic{
        Key<3>{std::array{-8, -6, 0}}, Key<3>{std::array{7, 5, 3}}, {true, true, true}};
    KeyBox<3> const bounded{Key<3>{std::array{-7, -6, -5}}, Key<3>{std::array{7, 6, 5}}, {}};
    for (int const r : {1, 3, 5})
    {
        auto const fp = refined_keys(periodic, r);
        EXPECT_NO_THROW(fp.check());
        EXPECT_EQ(fp.cells(), core::refine(periodic.cells(), r));
        for (auto const* keys : {&periodic, &bounded})
        {
            auto const fine = refined_keys(*keys, r);
            for (auto const site : {Site::corner, Site::body})
                core::for_each_point(core::grow(keys->cells(), 1), [&](auto const& c) {
                    auto const f = coincident_fine(core::Point<int, 3>{c}, site, r);
                    EXPECT_EQ(fine.has(f, site), keys->has(c, site)) << c;
                });
        }
    }
    auto const fb = refined_keys(bounded, 3);
    EXPECT_EQ(fb.lower, (Key<3>{std::array{-21, -18, -15}}));
    EXPECT_EQ(fb.upper, (Key<3>{std::array{21, 18, 15}}));
    EXPECT_THROW((void)refined_keys(bounded, 2), std::runtime_error);
}


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
    void do_refine(auto const& op, LevelRegion const region, LevelMap<3>& map,
                   std::vector<LevelPair> const& pairs, auto& coarse, auto& fine,
                   double const alpha)
    {
        refine(op, region, map, pairs, coarse, fine, alpha);
    }
    void do_coarsen(auto const& op, LevelMap<3>& map, std::vector<LevelPair> const& pairs,
                    auto& fine, auto& coarse)
    {
        coarsen(op, map, pairs, fine, coarse);
    }
} // namespace
} // namespace damn::amr::kelvin

#include "tst/damn/amr/kelvin_inter_level_tests.hpp" // IWYU pragma: keep


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
