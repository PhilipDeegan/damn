#ifndef DAMN_TST_AMR_KELVIN_INTER_LEVEL_TESTS_HPP
#define DAMN_TST_AMR_KELVIN_INTER_LEVEL_TESTS_HPP

// LinearRefine / InjectCoarsen through refine and coarsen on a two level Kelvin hierarchy, serial
// (test_kelvin_inter_level.cpp) and over MPI (test_mpi_kelvin_inter_level.cpp). The including file
// defines, before including this:
//   int test_rank(); int test_ranks(); Topology test_topology();
//   void do_refine(op, region, map, pairs, coarse_ld, fine_ld, alpha);
//   void do_coarsen(op, map, pairs, fine_ld, coarse_ld);

#include "damn/amr/kelvin/inter_level.hpp"
#include "damn/amr/kelvin/site_data.hpp"
#include "damn/amr/kelvin/sites.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/inter_level.hpp"
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
namespace
{
    using Box3   = core::Box<int, 3>;
    using Point3 = core::Point<int, 3>;
    using Pair   = std::array<double, 2>;

    struct PatchData
    {
        SiteField<double, 3> u;
        SiteField<Pair, 3> v;
    };

    void for_each_resource(PatchData& d, auto&& fn)
    {
        for_each_site_resource("u", d.u, fn);
        for_each_site_resource("v", d.v, fn);
    }

    // a linear field of a site's position (coarse h units: key / ratio^level)
    double f(Point3 const& cell, Site const site, int const ln)
    {
        auto const k   = key_of(cell, site);
        double const s = ln == 0 ? 1. : 1. / 3.;
        return 1.5 + 0.25 * k[0] * s - 0.5 * k[1] * s + 0.125 * k[2] * s;
    }

    double const nan = std::numeric_limits<double>::quiet_NaN();

    // coarse 8 x 6 x 4 cells (whole cells: every site, bounded), patches 4 3 2; fine: two patches
    //  side by side, a coarse cell clear of the domain's edge
    struct Levels
    {
        Hierarchy<3, Kelvin> h;
        LevelData<3, PatchData> coarse, fine;
        LevelMap<3> map;
    };

    PatchData make(Patch<3> const& p, int const ln)
    {
        PatchData d;
        d.u.allocate(p.box, nan);
        d.v.allocate(p.box, Pair{nan, nan});
        if (ln == 0)
            core::for_each_point(p.box, [&](auto const& c) {
                for (auto const site : {Site::corner, Site::body})
                {
                    value(d.u[site], p.box, c) = f(c, site, 0);
                    value(d.v[site], p.box, c) = Pair{f(c, site, 0), -2 * f(c, site, 0)};
                }
            });
        return d;
    }

    Levels make_levels()
    {
        Levels l;
        l.h.domain     = Box3{std::array{0, 0, 0}, std::array{7, 5, 3}};
        l.h.dx         = {1., 1., 1.};
        l.h.ratio      = 3;
        l.h.max_levels = 2;
        set_level(l.h, uniform_level_0(l.h, {4, 3, 2}, test_ranks()));
        // the fine patches on other ranks than the coarse ones under them
        set_level(
            l.h,
            make_level(1,
                       std::vector{core::refine(Box3{std::array{2, 1, 1}, std::array{4, 3, 2}}, 3),
                                   core::refine(Box3{std::array{5, 1, 1}, std::array{6, 3, 2}}, 3)},
                       std::vector{1 % test_ranks(), 2 % test_ranks()}));
        l.coarse = allocate(l.h.levels[0], test_rank(), [](auto const& p) { return make(p, 0); });
        l.fine   = allocate(l.h.levels[1], test_rank(), [](auto const& p) { return make(p, 1); });
        l.map    = make_level_map(l.h, 0, test_rank(), test_topology(), 6);
        return l;
    }

    std::vector<LevelPair> const pairs = same_level_names(site_names({"u", "v"}));

    // on a fine patch's domain (interior) or ghosts on no fine patch (level ghosts)
    bool in_region(Levels const& l, LevelRegion const region, Box3 const& box, Point3 const& c)
    {
        if (region == LevelRegion::Interior)
            return core::isIn(c, box);
        for (auto const& p : l.h.levels[1].patches)
            if (core::isIn(c, p.box))
                return false;
        return true;
    }
} // namespace

TEST(KelvinInterLevel, linearRefineIsExactForLinearFields)
{
    for (auto const region : {LevelRegion::Interior, LevelRegion::LevelGhosts})
    {
        auto l = make_levels();
        do_refine(LinearRefine{}, region, l.map, pairs, l.coarse, l.fine, 0.);
        std::size_t n = 0;
        for (std::size_t i = 0; i < size(l.fine); ++i)
        {
            auto const& box = l.fine.patches[i].box;
            auto& d         = l.fine.data[i];
            core::for_each_point(core::grow(box, 1), [&](auto const& c) {
                for (auto const site : {Site::corner, Site::body})
                {
                    auto const u  = value(d.u[site], box, c);
                    auto const& v = value(d.v[site], box, c);
                    if (!in_region(l, region, box, c))
                    {
                        EXPECT_TRUE(std::isnan(u)) << c;
                        continue;
                    }
                    ++n;
                    EXPECT_NEAR(u, f(c, site, 1), 1e-13) << c;
                    EXPECT_NEAR(v[1], -2 * f(c, site, 1), 1e-13) << c;
                }
            });
        }
        if (size(l.fine) > 0)
        {
            EXPECT_GT(n, 0u);
        }
    }
}

// in time with the coarse values at the step's end: an array per site, element by element
TEST(KelvinInterLevel, linearRefineInTime)
{
    auto l = make_levels();
    std::vector<LevelPair> const in_time{{"v.corner", "v.corner", "u.corner"}};
    for (std::size_t i = 0; i < size(l.coarse); ++i) // u.corner as the end's v: twice v
    {
        auto const& box = l.coarse.patches[i].box;
        core::for_each_point(box, [&](auto const& c) {
            value(l.coarse.data[i].v.corner, box, c) = Pair{f(c, Site::corner, 0), 0};
        });
    }
    std::vector<LevelPair> const pairs_v{{"v.corner", "v.corner", "v.corner"}};
    do_refine(LinearRefine{}, LevelRegion::Interior, l.map, pairs_v, l.coarse, l.fine, 0.25);
    for (std::size_t i = 0; i < size(l.fine); ++i)
    {
        auto const& box = l.fine.patches[i].box;
        core::for_each_point(box, [&](auto const& c) {
            auto const& v = value(l.fine.data[i].v.corner, box, c);
            EXPECT_NEAR(v[0], f(c, Site::corner, 1), 1e-13) << c;
            EXPECT_NEAR(v[1], 0, 1e-13) << c;
        });
    }
    // another resource of another kind at the step's end: refused
    if (size(l.fine) > 0)
    {
        EXPECT_THROW(do_refine(LinearRefine{}, LevelRegion::Interior, l.map, in_time, l.coarse,
                               l.fine, 0.25),
                     std::runtime_error);
    }
}

TEST(KelvinInterLevel, injectCoarsenTakesTheCoincidentFineSites)
{
    auto l = make_levels();
    for (std::size_t i = 0; i < size(l.fine); ++i) // fine: 10 + 3 f, every point
    {
        auto const& box = l.fine.patches[i].box;
        core::for_each_point(core::grow(box, 1), [&](auto const& c) {
            for (auto const site : {Site::corner, Site::body})
            {
                value(l.fine.data[i].u[site], box, c) = 10 + 3 * f(c, site, 1);
                value(l.fine.data[i].v[site], box, c) = Pair{0, 10 + 3 * f(c, site, 1)};
            }
        });
    }
    do_coarsen(InjectCoarsen{}, l.map, pairs, l.fine, l.coarse);

    std::size_t under = 0;
    for (std::size_t i = 0; i < size(l.coarse); ++i)
    {
        auto const& box = l.coarse.patches[i].box;
        core::for_each_point(box, [&](auto const& c) {
            bool covered = false;
            for (auto const& p : l.h.levels[1].patches)
                covered |= core::isIn(c, core::coarsen(p.box, 3));
            under += covered;
            for (auto const site : {Site::corner, Site::body})
            {
                auto const expect = covered ? 10 + 3 * f(c, site, 0) : f(c, site, 0);
                EXPECT_NEAR(value(l.coarse.data[i].u[site], box, c), expect, 1e-13) << c;
                EXPECT_NEAR(value(l.coarse.data[i].v[site], box, c)[1],
                            covered ? expect : -2 * f(c, site, 0), 1e-13)
                    << c;
            }
        });
    }
    if (test_ranks() == 1)
    {
        EXPECT_EQ(under, 5u * 3 * 2);
    }
}

} // namespace damn::amr::kelvin

#endif // DAMN_TST_AMR_KELVIN_INTER_LEVEL_TESTS_HPP
