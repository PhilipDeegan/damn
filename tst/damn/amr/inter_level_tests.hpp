#ifndef DAMN_TST_AMR_INTER_LEVEL_TESTS_HPP
#define DAMN_TST_AMR_INTER_LEVEL_TESTS_HPP

// refine and coarsen's tests with brute force expectations, serial (test_inter_level.cpp) and
// over MPI (test_mpi_inter_level.cpp). The including file defines, before including this:
//   int test_rank(); int test_ranks();
//   void do_refine(op, region, map, pairs, coarse_ld, fine_ld, alpha);
//   void do_coarsen(op, map, pairs, fine_ld, coarse_ld);
//   Topology test_topology();

#include "damn/amr/inter_level.hpp"
#include "damn/amr/level_map.hpp"
#include "tst/damn/amr/toy.hpp"

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace damn::amr
{
namespace
{
    // fine level: 16 x 12 cells. A and B side by side (their shared border is no level
    //  ghost), C at the upper corner (periodic images reach A)
    std::vector<Box2> const fine_boxes_{Box2{std::array{0, 0}, std::array{5, 3}},
                                        Box2{std::array{6, 0}, std::array{11, 3}},
                                        Box2{std::array{10, 6}, std::array{15, 11}}};

    Hierarchy<2> two_levels(std::array<bool, 2> const periodic)
    {
        auto h       = make_hierarchy(periodic, test_ranks());
        h.max_levels = 2;
        std::vector<int> ranks;
        for (std::size_t i = 0; i < fine_boxes_.size(); ++i) // not the coarse ones' ranks
            ranks.push_back(static_cast<int>(i + 1) % test_ranks());
        set_level(h, make_level(1, fine_boxes_, ranks));
        return h;
    }

    int constexpr level_reach = 6; // fine cells: toy fields' ghosts and stencils

    // values on level ln's points: periodic where the hierarchy is
    double f_on(Hierarchy<2> const& h, int const ln, Point2 p)
    {
        auto const shape = level_domain(h, ln).shape();
        for (std::size_t i = 0; i < 2; ++i)
            if (h.periodic[i])
                p[i] = mod(p[i], shape[i]);
        return 100. * p[0] + p[1];
    }

    std::vector<std::string> const field_names{"Bx", "rho", "Ez"};

    struct Levels
    {
        Hierarchy<2> h;
        LevelData_t coarse, fine;
        LevelMap<2> map;
    };

    // coarse fields f (protons_density) and 2 f (alpha_density) on their domain points, one
    //  particle per coarse domain cell; fine data NaN, no particles
    Levels make_levels(std::array<bool, 2> const periodic)
    {
        Levels l{two_levels(periodic), {}, {}, {}};
        l.coarse = allocate(l.h.levels[0], test_rank(), make_patch_data);
        l.fine   = allocate(l.h.levels[1], test_rank(), make_patch_data);
        for (std::size_t i = 0; i < size(l.coarse); ++i)
        {
            auto& d          = l.coarse.data[i];
            auto const& cell = l.coarse.patches[i].box;
            for (auto const& name : field_names)
            {
                auto& f = field(d, name);
                core::for_each_point(domain_box(f, cell),
                                     [&](auto const& p) { value(f, cell, p) = f_on(l.h, 0, p); });
            }
            core::for_each_point(domain_box(d.rho, cell), [&](auto const& p) {
                value(field(d, "protons_density"), cell, p) = f_on(l.h, 0, p);
                value(field(d, "alpha_density"), cell, p)   = 2 * f_on(l.h, 0, p);
            });
            core::for_each_point(cell, [&](auto const& p) {
                d.domain.items.push_back({{p[0], p[1]}, {p[0], p[1]}, l.coarse.patches[i].id});
            });
        }
        l.map = make_level_map(l.h, 0, test_rank(), test_topology(), level_reach);
        return l;
    }

    // fine point p from the coarse point under it
    struct Inject
    {
        int stencil = 0;

        void operator()(toy::Field& f, Box2 const& cells, Box2 const& box,
                        BoxValues<double const, 2> const& coarse) const
        {
            core::for_each_point(
                box, [&](auto const& p) { value(f, cells, p) = coarse(core::floor_div(p, 2)); });
        }

        // every coarse item split in one per fine cell of its coarse cell, those in box kept
        void operator()(toy::Particles& particles, Box2 const&, Box2 const& box,
                        std::span<toy::Item const> const coarse, Point2 const& shift) const
        {
            for (auto const& item : coarse)
                for (int x = 0; x < 2; ++x)
                    for (int y = 0; y < 2; ++y)
                    {
                        auto const c = cell(item) + shift;
                        Point2 const fine_cell{std::array{2 * c[0] + x, 2 * c[1] + y}};
                        if (core::isIn(fine_cell, box))
                        {
                            auto child = item;
                            child.cell = {fine_cell[0], fine_cell[1]};
                            append(particles, child, Point2{});
                        }
                    }
        }
    };

    // the fine points of region for a fine patch's resource (brute force)
    bool in_region(Levels const& l, LevelRegion const region, auto const& res, Box2 const& cells,
                   Point2 const& p)
    {
        if (region == LevelRegion::Interior)
            return core::isIn(p, domain_box(res, cells));
        if (!core::isIn(p, ghost_box(res, cells)))
            return false;
        std::vector<Point2> shifts{Point2{}};
        for (auto const& s : periodic_shifts(l.h, 1))
            shifts.push_back(s);
        for (auto const& g : l.h.levels[1].patches)
            for (auto const& s : shifts)
                if (core::isIn(p, domain_box(res, core::shift(g.box, s))))
                    return false;
        return true;
    }

    void expect_refined(Levels& l, LevelRegion const region, double const scale = 1)
    {
        for (std::size_t i = 0; i < size(l.fine); ++i)
            for (auto const& name : field_names)
            {
                auto& f          = field(l.fine.data[i], name);
                auto const& cell = l.fine.patches[i].box;
                std::size_t n    = 0;
                core::for_each_point(ghost_box(f, cell), [&](auto const& p) {
                    if (in_region(l, region, f, cell, p))
                    {
                        // level ghosts past a non periodic edge have no coarse data
                        if (!core::isIn(core::floor_div(p, 2), domain_box(f, l.h.domain))
                            and region == LevelRegion::LevelGhosts)
                            return;
                        ++n;
                        EXPECT_EQ(value(f, cell, p), scale * f_on(l.h, 0, core::floor_div(p, 2)))
                            << name << " " << i << " " << p;
                    }
                    else
                        EXPECT_TRUE(std::isnan(value(f, cell, p))) << name << " " << i << " " << p;
                });
                EXPECT_GT(n, 0u) << name << " " << i;
            }
    }
} // namespace


TEST(Refine, interiorFromTheCoarsePointsUnder)
{
    for (auto const periodic : {false, true})
    {
        auto l = make_levels({periodic, periodic});
        do_refine(Inject{}, LevelRegion::Interior, l.map, same_level_names(field_names), l.coarse,
                  l.fine, 0.);
        expect_refined(l, LevelRegion::Interior);
    }
}

TEST(Refine, levelGhostsOnlyWhereNoFinePatchIs)
{
    for (auto const periodic : {false, true})
    {
        auto l = make_levels({periodic, periodic});
        do_refine(Inject{}, LevelRegion::LevelGhosts, l.map, same_level_names(field_names),
                  l.coarse, l.fine, 0.);
        expect_refined(l, LevelRegion::LevelGhosts);
    }
}

TEST(Refine, timeInterpolated)
{
    auto l = make_levels({true, true});
    // rho from protons_density (f) and alpha_density (2 f): 1.25 f at alpha = 0.25
    do_refine(Inject{}, LevelRegion::Interior, l.map, {{"protons_density", "rho", "alpha_density"}},
              l.coarse, l.fine, 0.25);
    for (std::size_t i = 0; i < size(l.fine); ++i)
    {
        auto& f          = l.fine.data[i].rho;
        auto const& cell = l.fine.patches[i].box;
        core::for_each_point(domain_box(f, cell), [&](auto const& p) {
            EXPECT_DOUBLE_EQ(value(f, cell, p), 1.25 * f_on(l.h, 0, core::floor_div(p, 2))) << p;
        });
    }
}

TEST(Refine, theEndsOfATimeInterpolationReadOneSideOnly)
{
    for (double const alpha : {0., 1.})
    {
        auto l = make_levels({true, true});
        // the side alpha does not weigh is unset
        for (std::size_t i = 0; i < size(l.coarse); ++i)
        {
            auto& f = field(l.coarse.data[i], alpha == 1 ? "protons_density" : "alpha_density");
            std::fill(f.v.begin(), f.v.end(), nan);
        }
        do_refine(Inject{}, LevelRegion::Interior, l.map,
                  {{"protons_density", "rho", "alpha_density"}}, l.coarse, l.fine, alpha);
        for (std::size_t i = 0; i < size(l.fine); ++i)
        {
            auto& f          = l.fine.data[i].rho;
            auto const& cell = l.fine.patches[i].box;
            core::for_each_point(domain_box(f, cell), [&](auto const& p) {
                EXPECT_EQ(value(f, cell, p), (1 + alpha) * f_on(l.h, 0, core::floor_div(p, 2)))
                    << alpha << " " << p;
            });
        }
    }
}

TEST(Refine, particlesSplitIntoTheRegion)
{
    for (auto const region : {LevelRegion::Interior, LevelRegion::LevelGhosts})
    {
        auto l = make_levels({true, true});
        do_refine(Inject{}, region, l.map, {{"domain", "patch_ghost", ""}}, l.coarse, l.fine, 0.);
        for (std::size_t i = 0; i < size(l.fine); ++i)
        {
            auto const& particles = l.fine.data[i].patch_ghost;
            auto const& cell      = l.fine.patches[i].box;
            // one child per fine cell of the region
            std::vector<int> count(ghost_box(particles, cell).size(), 0);
            auto const gb = ghost_box(particles, cell);
            for (auto const& item : particles.items)
            {
                Point2 const c{item.cell};
                ASSERT_TRUE(in_region(l, region, particles, cell, c)) << c;
                ++count[static_cast<std::size_t>((c[1] - gb.lower[1]) * gb.shape()[0]
                                                 + (c[0] - gb.lower[0]))];
            }
            core::for_each_point(gb, [&](auto const& c) {
                auto const n = count[static_cast<std::size_t>((c[1] - gb.lower[1]) * gb.shape()[0]
                                                              + (c[0] - gb.lower[0]))];
                EXPECT_EQ(n, in_region(l, region, particles, cell, c) ? 1 : 0) << i << " " << c;
            });
        }
    }
}

TEST(Refine, theSecondCallReusesThePlan)
{
    auto l     = make_levels({true, true});
    auto names = same_level_names(field_names);
    do_refine(Inject{}, LevelRegion::LevelGhosts, l.map, names, l.coarse, l.fine, 0.);
    auto const plans = l.map.plans.size();
    auto const* plan = l.map.plans.begin()->second.get();
    do_refine(Inject{}, LevelRegion::LevelGhosts, l.map, names, l.coarse, l.fine, 0.);
    EXPECT_EQ(l.map.plans.size(), plans);
    EXPECT_EQ(l.map.plans.begin()->second.get(), plan);
    expect_refined(l, LevelRegion::LevelGhosts);
}

TEST(Refine, pastTheReachThrows)
{
    auto l = make_levels({true, true});
    l.map  = make_level_map(l.h, 0, test_rank(), test_topology(), 0);
    if (size(l.fine) == 0)
        GTEST_SKIP() << "no fine patch on this rank";
    EXPECT_THROW(do_refine(Inject{.stencil = 2}, LevelRegion::LevelGhosts, l.map,
                           same_level_names(field_names), l.coarse, l.fine, 0.),
                 std::runtime_error);
}


namespace
{
    // coarse point q from the fine point on it
    struct InjectCoarse
    {
        void operator()(toy::Field const& f, Box2 const& cells,
                        BoxValues<double, 2> const& out) const
        {
            core::for_each_point(out.box, [&](auto const& q) { out(q) = value(f, cells, q * 2); });
        }
    };
} // namespace

TEST(Coarsen, theCoarsePointsUnderTheFineDomains)
{
    for (auto const periodic : {false, true})
    {
        auto l = make_levels({periodic, periodic});
        for (std::size_t i = 0; i < size(l.fine); ++i)
            for (auto const& name : field_names)
            {
                auto& f          = field(l.fine.data[i], name);
                auto const& cell = l.fine.patches[i].box;
                core::for_each_point(ghost_box(f, cell),
                                     [&](auto const& p) { value(f, cell, p) = -f_on(l.h, 1, p); });
            }
        do_coarsen(InjectCoarse{}, l.map, same_level_names(field_names), l.fine, l.coarse);

        std::vector<Point2> shifts{Point2{}};
        for (auto const& s : periodic_shifts(l.h, 0))
            shifts.push_back(s);
        for (std::size_t i = 0; i < size(l.coarse); ++i)
            for (auto const& name : field_names)
            {
                auto& f          = field(l.coarse.data[i], name);
                auto const& cell = l.coarse.patches[i].box;
                core::for_each_point(domain_box(f, cell), [&](auto const& q) {
                    bool under = false;
                    for (auto const& g : l.h.levels[1].patches)
                        for (auto const& s : shifts)
                            under |= core::isIn(
                                q, domain_box(f, core::shift(core::coarsen(g.box, 2), s)));
                    EXPECT_EQ(value(f, cell, q), under ? -f_on(l.h, 1, q * 2) : f_on(l.h, 0, q))
                        << name << " " << i << " " << q;
                });
            }
    }
}

} // namespace damn::amr


// two patch data types, as two models' levels (one under the other): the fine level's its own
//  resource names, ghost widths and no particles, the pairs name both sides
namespace toy
{
struct OtherPatchData
{
    Field B_x{{true, false}, 3}, density{{true, true}, 1};
};
inline void for_each_resource(OtherPatchData& d, auto&& fn)
{
    fn("B_x", d.B_x);
    fn("density", d.density);
}
} // namespace toy

namespace damn::amr
{
namespace
{
    std::vector<LevelPair> const across{{"Bx", "B_x", ""}, {"rho", "density", ""}};

    toy::OtherPatchData make_other(Patch<2> const& patch)
    {
        toy::OtherPatchData d;
        d.B_x.v.assign(ghost_box(d.B_x, patch.box).size(), nan);
        d.density.v.assign(ghost_box(d.density, patch.box).size(), nan);
        return d;
    }
    toy::Field& other_field(toy::OtherPatchData& d, std::string const& name)
    {
        return name == "B_x" ? d.B_x : d.density;
    }
} // namespace

TEST(Refine, intoAnotherPatchDataType)
{
    for (auto const periodic : {false, true})
        for (auto const region : {LevelRegion::Interior, LevelRegion::LevelGhosts})
        {
            auto l     = make_levels({periodic, periodic});
            auto other = allocate(l.h.levels[1], test_rank(), make_other);
            do_refine(Inject{}, region, l.map, across, l.coarse, other, 0.);
            for (std::size_t i = 0; i < size(other); ++i)
                for (auto const& pair : across)
                {
                    auto& f          = other_field(other.data[i], pair.fine);
                    auto const& cell = other.patches[i].box;
                    std::size_t n    = 0;
                    core::for_each_point(ghost_box(f, cell), [&](auto const& p) {
                        if (in_region(l, region, f, cell, p))
                        {
                            if (!core::isIn(core::floor_div(p, 2), domain_box(f, l.h.domain))
                                and region == LevelRegion::LevelGhosts)
                                return;
                            ++n;
                            EXPECT_EQ(value(f, cell, p), f_on(l.h, 0, core::floor_div(p, 2)))
                                << pair.fine << " " << i << " " << p;
                        }
                        else
                            EXPECT_TRUE(std::isnan(value(f, cell, p)))
                                << pair.fine << " " << i << " " << p;
                    });
                    EXPECT_GT(n, 0u) << pair.fine << " " << i;
                }
        }
}

TEST(Coarsen, fromAnotherPatchDataType)
{
    for (auto const periodic : {false, true})
    {
        auto l     = make_levels({periodic, periodic});
        auto other = allocate(l.h.levels[1], test_rank(), make_other);
        for (std::size_t i = 0; i < size(other); ++i)
            for (auto const& pair : across)
            {
                auto& f          = other_field(other.data[i], pair.fine);
                auto const& cell = other.patches[i].box;
                core::for_each_point(ghost_box(f, cell),
                                     [&](auto const& p) { value(f, cell, p) = -f_on(l.h, 1, p); });
            }
        do_coarsen(InjectCoarse{}, l.map, across, other, l.coarse);

        std::vector<Point2> shifts{Point2{}};
        for (auto const& s : periodic_shifts(l.h, 0))
            shifts.push_back(s);
        for (std::size_t i = 0; i < size(l.coarse); ++i)
            for (auto const& pair : across)
            {
                auto& f          = field(l.coarse.data[i], pair.coarse);
                auto const& cell = l.coarse.patches[i].box;
                core::for_each_point(domain_box(f, cell), [&](auto const& q) {
                    bool under = false;
                    for (auto const& g : l.h.levels[1].patches)
                        for (auto const& s : shifts)
                            under |= core::isIn(
                                q, domain_box(f, core::shift(core::coarsen(g.box, 2), s)));
                    EXPECT_EQ(value(f, cell, q), under ? -f_on(l.h, 1, q * 2) : f_on(l.h, 0, q))
                        << pair.coarse << " " << i << " " << q;
                });
            }
    }
}

} // namespace damn::amr

#endif // DAMN_TST_AMR_INTER_LEVEL_TESTS_HPP
