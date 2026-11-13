// damn::amr same level schedules on a toy field: ghost fills (periodic or not), border sum
// and max, and the Scheduler's named, merged and rebuilt schedules
// expected values come from brute force over every patch and periodic image

#include "damn/amr/scheduler.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace toy
{
using Box2   = damn::core::Box<int, 2>;
using Point2 = damn::core::Point<int, 2>;

// values on the nodes of a patch: primal nodes on cell corners (one more than cells),
// dual on cell centres, gw ghost nodes each side
struct Field
{
    std::array<bool, 2> primal{};
    int gw = 2;
    std::vector<double> v{};
};

inline Box2 domain_box(Field const& f, Box2 cells)
{
    for (std::size_t i = 0; i < 2; ++i)
        cells.upper[i] += f.primal[i];
    return cells;
}

inline Box2 ghost_box(Field const& f, Box2 const& cells)
{
    return damn::core::grow(domain_box(f, cells), f.gw);
}

inline std::size_t flat(Field const& f, Box2 const& cells, Point2 const& p)
{
    auto const gb = ghost_box(f, cells);
    auto const nx = gb.upper[0] - gb.lower[0] + 1;
    assert(damn::core::isIn(p, gb));
    return (p[1] - gb.lower[1]) * nx + (p[0] - gb.lower[0]);
}

inline double& value(Field& f, Box2 const& cells, Point2 const& p)
{
    return f.v[flat(f, cells, p)];
}
inline double const& value(Field const& f, Box2 const& cells, Point2 const& p)
{
    return f.v[flat(f, cells, p)];
}

struct Population
{
    std::string name;
    Field density;
    std::vector<int> particles; // not BoxData
};

struct PatchData
{
    Field Bx{{true, false}}, rho{{true, true}}, Ez{{false, false}};
    std::vector<Population> populations;
};

inline void for_each_resource(PatchData& data, auto&& fn)
{
    fn("Bx", data.Bx);
    fn("rho", data.rho);
    fn("Ez", data.Ez);
    for (auto& pop : data.populations)
    {
        fn(pop.name + "_density", pop.density);
        fn(pop.name + "_particles", pop.particles);
    }
}
} // namespace toy


namespace damn::amr
{
static_assert(BoxData<toy::Field, 2>);
static_assert(not BoxData<std::vector<int>, 2>);

namespace
{
    using toy::Box2;
    using toy::Point2;
    using LevelData_t = LevelData<2, toy::PatchData>;

    double const nan = std::numeric_limits<double>::quiet_NaN();

    // 8 x 6 cells, patches of at most 3 x 4: 3 x 2 of them, uneven
    Hierarchy<2> make_hierarchy(std::array<bool, 2> const periodic, int const n_ranks = 1)
    {
        Hierarchy<2> h;
        h.domain   = Box2{std::array{0, 0}, std::array{7, 5}};
        h.dx       = {.1, .1};
        h.periodic = periodic;
        set_level(h, uniform_level_0(h, {3, 4}, n_ranks));
        return h;
    }

    toy::PatchData make_patch_data(Patch<2> const& patch)
    {
        toy::PatchData data;
        data.populations = {{"protons", {{true, true}}, {}}, {"alpha", {{true, true}}, {}}};
        auto const init  = [&](toy::Field& f) {
            f.v.assign(ghost_box(f, patch.box).size(), nan);
        };
        init(data.Bx);
        init(data.rho);
        init(data.Ez);
        for (auto& pop : data.populations)
            init(pop.density);
        return data;
    }

    toy::Field& field(toy::PatchData& d, std::string const& name)
    {
        toy::Field* f = nullptr;
        for_each_resource(d, [&](std::string_view n, auto& r) {
            if constexpr (std::is_same_v<std::decay_t<decltype(r)>, toy::Field>)
                if (n == name)
                    f = &r;
        });
        assert(f);
        return *f;
    }

    int mod(int a, int n) { return ((a % n) + n) % n; }

    // a function of the physical point: periodic where the hierarchy is
    double f_of(Hierarchy<2> const& h, Point2 p)
    {
        auto const shape = level_domain(h, 0).shape();
        for (std::size_t i = 0; i < 2; ++i)
            if (h.periodic[i])
                p[i] = mod(p[i], shape[i]);
        return 100. * p[0] + p[1];
    }

    // every (patch, shift) image of a field's box, for brute force
    template<typename BoxFn>
    std::vector<std::pair<Patch<2>, Box2>> images(Hierarchy<2> const& h, BoxFn&& box_of)
    {
        std::vector<Point2> shifts{Point2{}};
        for (auto const& s : periodic_shifts(h, 0))
            shifts.push_back(s);
        std::vector<std::pair<Patch<2>, Box2>> out;
        for (auto const& p : h.levels[0].patches)
            for (auto const& s : shifts)
                out.emplace_back(p, core::shift(box_of(p), s));
        return out;
    }
} // namespace


// ---------------------------------------------------------------- PatchGhost

class GhostFill : public ::testing::TestWithParam<std::tuple<std::string, bool>>
{
};

TEST_P(GhostFill, ghostsGetTheNeighboursDomainValues)
{
    auto const& [name, periodic] = GetParam();
    auto const h                 = make_hierarchy({periodic, periodic});
    auto ld                      = allocate(h.levels[0], 0, make_patch_data);

    for_each_patch(ld, [&](auto const& patch, auto& data) {
        auto& f = field(data, name);
        core::for_each_point(domain_box(f, patch.box),
                             [&](auto const& p) { value(f, patch.box, p) = f_of(h, p); });
    });

    auto const schedule = make_schedule(Transfer::PatchGhost, h, 0, ld, 0, {name});
    run(schedule, ld);

    // a ghost point is filled iff some (other or shifted) patch's domain covers it
    auto const domain_nodes = [&](auto const& f, auto const& p) { return domain_box(f, p.box); };
    std::size_t n_filled = 0, n_unfilled = 0;
    for_each_patch(ld, [&](auto const& patch, auto& data) {
        auto& f         = field(data, name);
        auto const imgs = images(h, [&](auto const& p) { return domain_nodes(f, p); });
        core::for_each_point(ghost_box(f, patch.box), [&](auto const& p) {
            bool covered = false;
            for (auto const& [_, b] : imgs)
                covered = covered or isIn(p, b);
            auto const v = value(f, patch.box, p);
            if (covered)
            {
                EXPECT_EQ(v, f_of(h, p)) << name << " patch " << patch.id << " " << p;
                ++n_filled;
            }
            else
            {
                EXPECT_TRUE(std::isnan(v)) << name << " patch " << patch.id << " " << p;
                ++n_unfilled;
            }
        });
    });

    EXPECT_GT(n_filled, 0u);
    if (periodic)
        EXPECT_EQ(n_unfilled, 0u); // everything has a neighbour
    else
        EXPECT_GT(n_unfilled, 0u); // outside the domain
}

INSTANTIATE_TEST_SUITE_P(Schedule, GhostFill,
                         ::testing::Combine(::testing::Values("Bx", "rho", "Ez",
                                                              "alpha_density"),
                                            ::testing::Bool()));


// ---------------------------------------------------------------- BorderSum / BorderMax

class BorderOps : public ::testing::TestWithParam<bool> // periodic
{
};

TEST_P(BorderOps, sumCountsEveryPatchImageCoveringAPoint)
{
    auto const h = make_hierarchy({GetParam(), GetParam()});
    auto ld      = allocate(h.levels[0], 0, make_patch_data);

    for_each_patch(ld, [&](auto const& patch, auto& data) {
        auto& f = data.rho;
        core::for_each_point(ghost_box(f, patch.box), [&](auto const& p) { value(f, patch.box, p) = 1; });
    });

    run(make_schedule(Transfer::BorderSum, h, 0, ld, 0, {"rho"}), ld);

    for_each_patch(ld, [&](auto const& patch, auto& data) {
        auto& f         = data.rho;
        auto const imgs = images(h, [&](auto const& p) { return ghost_box(f, p.box); });
        core::for_each_point(ghost_box(f, patch.box), [&](auto const& p) {
            double count = 0;
            for (auto const& [_, b] : imgs)
                count += isIn(p, b);
            EXPECT_EQ(value(f, patch.box, p), count) << "patch " << patch.id << " " << p;
        });
    });
}

TEST_P(BorderOps, maxIsTheLargestOfEveryPatchImageCoveringAPoint)
{
    auto const h = make_hierarchy({GetParam(), GetParam()});
    auto ld      = allocate(h.levels[0], 0, make_patch_data);

    for_each_patch(ld, [&](auto const& patch, auto& data) {
        auto& f = data.Bx;
        core::for_each_point(ghost_box(f, patch.box),
                             [&](auto const& p) { value(f, patch.box, p) = patch.id + 1.; });
    });

    run(make_schedule(Transfer::BorderMax, h, 0, ld, 0, {"Bx"}), ld);

    for_each_patch(ld, [&](auto const& patch, auto& data) {
        auto& f         = data.Bx;
        auto const imgs = images(h, [&](auto const& p) { return ghost_box(f, p.box); });
        core::for_each_point(ghost_box(f, patch.box), [&](auto const& p) {
            double max = 0;
            for (auto const& [img, b] : imgs)
                if (isIn(p, b))
                    max = std::max(max, img.id + 1.);
            EXPECT_EQ(value(f, patch.box, p), max) << "patch " << patch.id << " " << p;
        });
    });
}

INSTANTIATE_TEST_SUITE_P(Schedule, BorderOps, ::testing::Bool());


// ---------------------------------------------------------------- Scheduler

TEST(Scheduler, oneKeyExchangesAllItsResources)
{
    auto const h = make_hierarchy({true, true});
    auto ld      = allocate(h.levels[0], 0, make_patch_data);

    Scheduler<2> scheduler;
    add(scheduler, "moments", Transfer::PatchGhost, {"rho"});
    add(scheduler, "moments", Transfer::PatchGhost, {"protons_density", "alpha_density"});
    EXPECT_EQ(names(scheduler, "moments", Transfer::PatchGhost).size(), 3u);
    EXPECT_FALSE(has(scheduler, "moments", Transfer::BorderSum));

    for_each_patch(ld, [&](auto const& patch, auto& data) {
        for (auto const* name : {"rho", "protons_density", "alpha_density"})
        {
            auto& f = field(data, name);
            core::for_each_point(domain_box(f, patch.box),
                                 [&](auto const& p) { value(f, patch.box, p) = f_of(h, p); });
        }
    });

    fill<Transfer::PatchGhost>(scheduler, "moments", h, ld, 0);

    for_each_patch(ld, [&](auto const& patch, auto& data) {
        for (auto const* name : {"rho", "protons_density", "alpha_density"})
        {
            auto& f = field(data, name);
            core::for_each_point(ghost_box(f, patch.box), [&](auto const& p) {
                EXPECT_EQ(value(f, patch.box, p), f_of(h, p)) << name;
            });
        }
    });
}

TEST(Scheduler, rebuildsTheScheduleWhenTheLevelChanges)
{
    auto h  = make_hierarchy({true, true});
    auto ld = allocate(h.levels[0], 0, make_patch_data);

    Scheduler<2> scheduler;
    add(scheduler, "B", Transfer::PatchGhost, {"Bx"});

    auto const* first = &schedule_for<Transfer::PatchGhost>(scheduler, "B", h, ld, 0);
    EXPECT_EQ(first->built_on.size(), 6u);
    EXPECT_EQ(&schedule_for<Transfer::PatchGhost>(scheduler, "B", h, ld, 0), first); // cached

    set_level(h, uniform_level_0(h, {4, 6})); // regrid level 0: 2 patches
    ld = allocate(h.levels[0], 0, make_patch_data);

    auto const& rebuilt = schedule_for<Transfer::PatchGhost>(scheduler, "B", h, ld, 0);
    EXPECT_EQ(rebuilt.built_on.size(), 2u);
    EXPECT_TRUE(is_current(rebuilt, h.levels[0]));
}

TEST(Scheduler, rejectsUnknownAndNonBoxResources)
{
    auto const h = make_hierarchy({false, false});
    auto ld      = allocate(h.levels[0], 0, make_patch_data);

    Scheduler<2> scheduler;
    EXPECT_THROW(fill<Transfer::PatchGhost>(scheduler, "none", h, ld, 0), std::runtime_error);

    add(scheduler, "parts", Transfer::PatchGhost, {"alpha_particles"});
    EXPECT_THROW(fill<Transfer::PatchGhost>(scheduler, "parts", h, ld, 0), std::runtime_error);

    add(scheduler, "typo", Transfer::PatchGhost, {"rhoo"});
    EXPECT_THROW(fill<Transfer::PatchGhost>(scheduler, "typo", h, ld, 0), std::runtime_error);
}

TEST(Scheduler, remoteSourcesAreNotDoneYet)
{
    auto const h = make_hierarchy({true, true}, 2); // patches split over 2 ranks
    auto ld      = allocate(h.levels[0], 0, make_patch_data);

    auto const schedule = make_schedule(Transfer::PatchGhost, h, 0, ld, 0, {"rho"});
    EXPECT_THROW(run(schedule, ld), std::runtime_error);
}

} // namespace damn::amr


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
