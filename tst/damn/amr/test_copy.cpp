// damn::amr same level copies on a toy field: ghost fills (periodic or not), border sum and
// max, leaving particles; the Neighbours' links and distances, plans kept and dropped, and
// the LevelMap between two levels
// expected values come from brute force over every patch and periodic image

#include "damn/amr/level_map.hpp"
#include "damn/amr/copy.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cstdint>
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
    std::vector<int> particles; // neither BoxData nor ParticleData
};

// a particle: its cell and where it started, for checks
struct Item
{
    std::array<int, 2> cell;
    std::array<int, 2> origin;
    std::uint32_t from_patch;
};

struct Particles
{
    using value_type = Item;
    int gw           = 1; // ghost cells a patch may hold particles in
    std::vector<Item> items{};
};

inline Box2 domain_box(Particles const&, Box2 const& cells)
{
    return cells;
}
inline Box2 ghost_box(Particles const& p, Box2 const& cells)
{
    return damn::core::grow(cells, p.gw);
}
inline void for_each_item(Particles const& p, auto&& fn)
{
    for (auto const& item : p.items)
        fn(item);
}
inline Point2 cell(Item const& item)
{
    return Point2{item.cell};
}
inline void append(Particles& p, Item item, Point2 const& shift)
{
    for (std::size_t i = 0; i < 2; ++i)
        item.cell[i] += shift[i];
    p.items.push_back(item);
}

struct PatchData
{
    Field Bx{{true, false}}, rho{{true, true}}, Ez{{false, false}};
    std::vector<Population> populations;
    Particles domain, patch_ghost;
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
    fn("domain", data.domain);
    fn("patch_ghost", data.patch_ghost);
}
} // namespace toy


namespace damn::amr
{
static_assert(BoxData<toy::Field, 2>);
static_assert(not BoxData<std::vector<int>, 2>);
static_assert(ParticleData<toy::Particles, 2>);
static_assert(not ParticleData<std::vector<int>, 2>);
static_assert(not BoxData<toy::Particles, 2>);

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

    int constexpr reach = 3; // toy fields: 2 ghosts, plus one for primal nodes

    Neighbours<2> neighbours_of(Hierarchy<2> const& h, int const rank = 0,
                                Topology const& t = single_node())
    {
        return make_neighbours(h, 0, rank, t, reach);
    }

    toy::PatchData make_patch_data(Patch<2> const& patch)
    {
        toy::PatchData data;
        data.populations = {{"protons", {{true, true}}, {}}, {"alpha", {{true, true}}, {}}};
        auto const init  = [&](toy::Field& f) { f.v.assign(ghost_box(f, patch.box).size(), nan); };
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

    int mod(int a, int n)
    {
        return ((a % n) + n) % n;
    }

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

    auto n = neighbours_of(h);
    copy(fill_ghosts, n, {name}, ld);

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

INSTANTIATE_TEST_SUITE_P(Copy, GhostFill,
                         ::testing::Combine(::testing::Values("Bx", "rho", "Ez", "alpha_density"),
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
        core::for_each_point(ghost_box(f, patch.box),
                             [&](auto const& p) { value(f, patch.box, p) = 1; });
    });

    auto n = neighbours_of(h);
    copy(sum_borders, n, {"rho"}, ld);

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

    auto n = neighbours_of(h);
    copy(max_borders, n, {"Bx"}, ld);

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

INSTANTIATE_TEST_SUITE_P(Copy, BorderOps, ::testing::Bool());


// ---------------------------------------------------------------- GhostToDomain

class GhostParticles : public ::testing::TestWithParam<bool> // periodic
{
};

TEST_P(GhostParticles, leaveThePatchIntoTheDomainOwningTheirCell)
{
    bool const periodic = GetParam();
    auto const h        = make_hierarchy({periodic, periodic});
    auto ld             = allocate(h.levels[0], 0, make_patch_data);

    // one particle in each ghost cell of each patch, as if it had just left the patch
    std::size_t n_left = 0;
    for_each_patch(ld, [&](auto const& patch, auto& data) {
        core::for_each_point(ghost_box(data.patch_ghost, patch.box), [&](auto const& c) {
            if (!isIn(c, patch.box))
            {
                data.patch_ghost.items.push_back({{c[0], c[1]}, {c[0], c[1]}, patch.id});
                ++n_left;
            }
        });
    });

    auto n = neighbours_of(h);
    copy(gather_leaving, n, std::vector<ResourcePair>{{"patch_ghost", "domain"}}, ld);

    auto const level_cells = level_domain(h, 0);
    auto const shape       = level_cells.shape();
    auto const wrap        = [&](std::array<int, 2> c) {
        for (std::size_t i = 0; i < 2; ++i)
            if (periodic)
                c[i] = mod(c[i], shape[i]);
        return c;
    };

    // each particle arrived exactly where its (wrapped) cell is owned
    std::size_t n_arrived = 0, n_inside = 0;
    for_each_patch(ld, [&](auto const& patch, auto& data) {
        for (auto const& item : data.domain.items)
        {
            ++n_arrived;
            EXPECT_TRUE(isIn(Point2{item.cell}, patch.box)) << patch.id;
            EXPECT_EQ(item.cell, wrap(item.origin));
            // never handed back to the patch it left, unless through a periodic image
            EXPECT_FALSE(item.from_patch == patch.id and item.cell == item.origin);
        }
        for (auto const& item : data.patch_ghost.items)
            n_inside += isIn(Point2{item.origin}, level_cells);
    });

    if (periodic)
        EXPECT_EQ(n_arrived, n_left); // every cell has an owner
    else
        EXPECT_EQ(n_arrived, n_inside); // those that left the domain are dropped
    EXPECT_GT(n_arrived, 0u);
}

INSTANTIATE_TEST_SUITE_P(Copy, GhostParticles, ::testing::Bool());

TEST(Copy, leavingNeedsParticleData)
{
    auto const h = make_hierarchy({true, true});
    auto ld      = allocate(h.levels[0], 0, make_patch_data);
    auto n       = neighbours_of(h);
    EXPECT_THROW(copy(gather_leaving, n, std::vector<ResourcePair>{{"rho", "rho"}}, ld),
                 std::runtime_error);
}

TEST(Copy, oneCopyExchangesAllItsResources)
{
    auto const h = make_hierarchy({true, true});
    auto ld      = allocate(h.levels[0], 0, make_patch_data);
    auto n       = neighbours_of(h);
    std::vector<std::string> const moments{"rho", "protons_density", "alpha_density"};

    for_each_patch(ld, [&](auto const& patch, auto& data) {
        for (auto const& name : moments)
        {
            auto& f = field(data, name);
            core::for_each_point(domain_box(f, patch.box),
                                 [&](auto const& p) { value(f, patch.box, p) = f_of(h, p); });
        }
    });

    copy(fill_ghosts, n, moments, ld);

    for_each_patch(ld, [&](auto const& patch, auto& data) {
        for (auto const& name : moments)
        {
            auto& f = field(data, name);
            core::for_each_point(ghost_box(f, patch.box), [&](auto const& p) {
                EXPECT_EQ(value(f, patch.box, p), f_of(h, p)) << name;
            });
        }
    });
}

TEST(Copy, rejectsUnknownAndNonBoxResources)
{
    auto const h = make_hierarchy({false, false});
    auto ld      = allocate(h.levels[0], 0, make_patch_data);
    auto n       = neighbours_of(h);

    EXPECT_THROW(copy(fill_ghosts, n, {"alpha_particles"}, ld), std::runtime_error);
    EXPECT_THROW(copy(fill_ghosts, n, {"rhoo"}, ld), std::runtime_error);
    EXPECT_THROW(copy(fill_ghosts, n, {"domain"}, ld), std::runtime_error); // ParticleData
}

TEST(Copy, plansAreMadeOnceAndKeptInTheNeighbours)
{
    auto const h = make_hierarchy({true, true});
    auto ld      = allocate(h.levels[0], 0, make_patch_data);
    auto n       = neighbours_of(h);

    copy(fill_ghosts, n, {"Bx"}, ld);
    copy(sum_borders, n, {"rho"}, ld);
    ASSERT_EQ(n.plans.size(), 2u);
    auto const* plan = &n.plans.begin()->second;

    copy(fill_ghosts, n, {"Bx"}, ld);
    EXPECT_EQ(n.plans.size(), 2u);
    EXPECT_EQ(&n.plans.begin()->second, plan);

    // the border sum's buffer holds every copy, no more
    auto const& sum   = plan_for(Region::Borders, n, ld, same_names({"rho"}));
    std::size_t bytes = 0;
    for (auto const& c : sum.pieces)
        bytes += c.bytes;
    EXPECT_EQ(sum.buffer.size(), bytes);
    EXPECT_GT(bytes, 0u);
}

TEST(Copy, aRegridNeedsNewNeighbours)
{
    auto h  = make_hierarchy({true, true});
    auto ld = allocate(h.levels[0], 0, make_patch_data);
    auto n  = neighbours_of(h);
    EXPECT_TRUE(is_current(n, h.levels[0]));
    EXPECT_EQ(n.local.size(), 6u);

    set_level(h, uniform_level_0(h, {4, 6})); // regrid level 0: 2 patches
    ld = allocate(h.levels[0], 0, make_patch_data);

    EXPECT_FALSE(is_current(n, h.levels[0]));
    EXPECT_THROW(copy(fill_ghosts, n, {"Bx"}, ld), std::runtime_error);

    n = neighbours_of(h);
    EXPECT_TRUE(is_current(n, h.levels[0]));
    EXPECT_NO_THROW(copy(fill_ghosts, n, {"Bx"}, ld));
}

TEST(Copy, theReachMustCoverEveryResource)
{
    auto const h = make_hierarchy({true, true});
    auto ld      = allocate(h.levels[0], 0, make_patch_data);
    auto short_n = make_neighbours(h, 0, 0, single_node(), reach - 1);
    EXPECT_THROW(copy(fill_ghosts, short_n, {"Bx"}, ld), std::runtime_error);
}

TEST(Copy, otherRanksAreNotDoneYet)
{
    auto const h = make_hierarchy({true, true}, 2); // patches split over 2 ranks
    auto ld      = allocate(h.levels[0], 0, make_patch_data);
    auto n       = neighbours_of(h, 0, single_node(2));
    EXPECT_EQ(farthest(n), Distance::Node);
    EXPECT_THROW(copy(fill_ghosts, n, {"rho"}, ld), std::runtime_error);
}


// ---------------------------------------------------------------- Topology, Neighbours

TEST(Topology, distancesAreOrdered)
{
    Location const a{0, 0, 0, -1};
    EXPECT_EQ(distance(a, a), Distance::Local);
    EXPECT_EQ(distance(a, Location{0, 0, 1, -1}), Distance::Numa); // other NUMA node
    EXPECT_EQ(distance(a, Location{0, 0, 0, 1}), Distance::Numa);  // a device
    EXPECT_EQ(distance(a, Location{1, 0, 0, -1}), Distance::Node);
    EXPECT_EQ(distance(a, Location{1, 1, 0, -1}), Distance::Remote);
    EXPECT_LT(Distance::Self, Distance::Local);
    EXPECT_LT(Distance::Local, Distance::Numa);
    EXPECT_LT(Distance::Numa, Distance::Node);
    EXPECT_LT(Distance::Node, Distance::Remote);
    EXPECT_THROW(std::ignore = location(single_node(2), 2), std::runtime_error);
}

TEST(Neighbours, linksCarryTheirDistance)
{
    auto const h = make_hierarchy({true, true}, 2);

    Topology two_nodes      = single_node(2);
    two_nodes.ranks[1].node = 1;

    for (auto const& [t, other] :
         {std::pair{single_node(2), Distance::Node}, std::pair{two_nodes, Distance::Remote}})
    {
        auto const n       = neighbours_of(h, 0, t);
        std::size_t n_self = 0, n_local = 0, n_other = 0;
        for (std::size_t i = 0; i < n.local.size(); ++i)
            for (auto const& link : n.of[i])
            {
                if (link.patch.id == n.local[i].id)
                {
                    EXPECT_EQ(link.distance, Distance::Self);
                    EXPECT_NE(link.shift, Point2{});
                    ++n_self;
                }
                else if (link.patch.rank == 0)
                {
                    EXPECT_EQ(link.distance, Distance::Local);
                    EXPECT_NE(link.local, not_local);
                    ++n_local;
                }
                else
                {
                    EXPECT_EQ(link.distance, other);
                    EXPECT_EQ(link.local, not_local);
                    ++n_other;
                }
            }
        EXPECT_GT(n_self, 0u); // periodic images of themselves
        EXPECT_GT(n_local, 0u);
        EXPECT_GT(n_other, 0u);
    }
}

TEST(Neighbours, onlyPatchesWithinReach)
{
    // 12 x 2 cells, 2 x 2 patches, not periodic: patch i only meets i - 1 and i + 1 at reach 1
    Hierarchy<2> h;
    h.domain = Box2{std::array{0, 0}, std::array{11, 1}};
    h.dx     = {.1, .1};
    set_level(h, uniform_level_0(h, {2, 2}));
    auto const n = make_neighbours(h, 0, 0, single_node(), 1);
    ASSERT_EQ(n.local.size(), 6u);
    for (std::size_t i = 0; i < n.local.size(); ++i)
    {
        EXPECT_EQ(n.of[i].size(), (i == 0 or i == 5) ? 1u : 2u) << i;
        for (auto const& link : n.of[i])
            EXPECT_EQ(std::abs(static_cast<int>(link.patch.id) - static_cast<int>(n.local[i].id)),
                      1);
    }
}


// ---------------------------------------------------------------- LevelMap

TEST(LevelMap, coarseAndFinePatchesAtTheSameLocation)
{
    auto h       = make_hierarchy({false, false});
    h.max_levels = 2;
    // level 1: two fine patches over coarse cells [1, 4] x [1, 2] and [5, 6] x [3, 4]
    set_level(h, make_level(1, std::vector{Box2{std::array{2, 2}, std::array{9, 5}},
                                           Box2{std::array{10, 6}, std::array{13, 9}}}));

    auto const map = make_level_map(h, 0, 0, single_node(), 0);
    ASSERT_EQ(map.local_fine.size(), 2u);
    ASSERT_EQ(map.local_coarse.size(), 6u);

    // brute force: coarse patch c is under fine patch f iff their coarse footprints meet
    for (std::size_t i = 0; i < map.local_fine.size(); ++i)
    {
        auto const footprint = core::coarsen(map.local_fine[i].box, h.ratio);
        std::size_t expected = 0;
        for (auto const& c : h.levels[0].patches)
            expected += static_cast<bool>(footprint * c.box);
        EXPECT_EQ(map.under[i].size(), expected) << i;
        for (auto const& link : map.under[i])
        {
            EXPECT_TRUE((footprint * link.patch.box).has_value());
            EXPECT_EQ(link.distance, Distance::Local);
        }
    }
    std::size_t n_over = 0;
    for (auto const& links : map.over)
        n_over += links.size();
    std::size_t n_under = 0;
    for (auto const& links : map.under)
        n_under += links.size();
    EXPECT_EQ(n_over, n_under); // the same overlaps, seen from either side
    EXPECT_TRUE(is_current(map, h));
}

} // namespace damn::amr


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
