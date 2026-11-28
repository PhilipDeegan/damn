// the serial copy tests' brute force expectations with the patches spread over the ranks (round
// robin, uniform_level_0); any number of ranks, one checks the MPI path against the serial copies

#include "damn/mpi/copy.hpp"
#include "damn/core/exec/thread_pool.hpp"
#include "damn/mpi/topology.hpp"
#include "damn/mpi/damn_mpi.hpp"
#include "tst/damn/amr/toy.hpp"
#include "damn/core/def/damn_config.hpp"
#include "damn/core/exec/kernels.hpp" // IWYU pragma: keep (gpu::Vec_t: GPU builds)
#include "damn/core/memory/box_span.hpp"

#include "gtest/gtest.h"

#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <algorithm>
#include <array>
#include <bit>
#include <stdexcept>
#include <cstdint>
#include <vector>
#include <type_traits>
#include <stdexcept>
#include <algorithm>

namespace damn::amr
{
namespace
{
    int mpi_rank()
    {
        int r = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &r);
        return r;
    }

    int mpi_size()
    {
        int s = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &s);
        return s;
    }

    std::size_t sum_over_ranks(std::size_t const local)
    {
        unsigned long long l = local, total = 0;
        MPI_Allreduce(&l, &total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        return static_cast<std::size_t>(total);
    }

    struct World
    {
        explicit World(bool const periodic)
            : h{make_hierarchy({periodic, periodic}, mpi_size())}
            , ld{allocate(h.levels[0], mpi_rank(), make_patch_data)}
            , n{make_neighbours(h, 0, mpi_rank(), mpi::topology(), reach)}
        {
        }

        Hierarchy<2> h;
        LevelData_t ld;
        Neighbours<2> n;
    };
} // namespace


TEST(MpiTopology, everyRankHasALocation)
{
    auto const t = mpi::topology();
    ASSERT_EQ(static_cast<int>(t.ranks.size()), mpi_size());
    for (int r = 0; r < mpi_size(); ++r)
        EXPECT_EQ(t.ranks[r].rank, r);
    EXPECT_EQ(distance(t, mpi_rank(), mpi_rank()), Distance::Local);
}


class MpiGhostFill : public ::testing::TestWithParam<std::tuple<std::string, bool>>
{
};

TEST_P(MpiGhostFill, ghostsGetTheNeighboursDomainValues)
{
    auto const& [name, periodic] = GetParam();
    World s{periodic};

    for_each_patch(s.ld, [&](auto const& patch, auto& data) {
        auto& f = field(data, name);
        core::for_each_point(domain_box(f, patch.box),
                             [&](auto const& p) { value(f, patch.box, p) = f_of(s.h, p); });
    });

    mpi::copy(fill_ghosts, s.n, {name}, s.ld);

    std::size_t n_filled = 0;
    for_each_patch(s.ld, [&](auto const& patch, auto& data) {
        auto& f         = field(data, name);
        auto const imgs = images(s.h, [&](auto const& p) { return domain_box(f, p.box); });
        core::for_each_point(ghost_box(f, patch.box), [&](auto const& p) {
            bool covered = false;
            for (auto const& [_, b] : imgs)
                covered = covered or isIn(p, b);
            auto const v = value(f, patch.box, p);
            if (covered)
            {
                EXPECT_EQ(v, f_of(s.h, p)) << name << " patch " << patch.id << " " << p;
                ++n_filled;
            }
            else
                EXPECT_TRUE(std::isnan(v)) << name << " patch " << patch.id << " " << p;
        });
    });
    EXPECT_GT(sum_over_ranks(n_filled), 0u);
}

// each patch's copy of a shared node different: the ghosts are those of one rank holding every
//  patch (each ghost point filled by the same image whatever the ranks)
TEST_P(MpiGhostFill, theRanksDoNotMatter)
{
    auto const& [name, periodic] = GetParam();
    auto const set               = [&](auto const& h, auto& ld) {
        for_each_patch(ld, [&](auto const& patch, auto& data) {
            auto& f = field(data, name);
            core::for_each_point(domain_box(f, patch.box), [&](auto const& p) {
                value(f, patch.box, p) = f_of(h, p) + 1e-3 * (patch.id + 1);
            });
        });
    };

    World s{periodic};
    set(s.h, s.ld);
    mpi::copy(fill_ghosts, s.n, {name}, s.ld);

    auto const h1 = make_hierarchy({periodic, periodic}); // every patch on this rank
    auto one      = allocate(h1.levels[0], 0, make_patch_data);
    auto n1       = neighbours_of(h1);
    set(h1, one);
    copy(fill_ghosts, n1, {name}, one);

    for (std::size_t i = 0; i < size(s.ld); ++i)
    {
        auto const& patch = s.ld.patches[i];
        auto const j      = static_cast<std::size_t>(
            std::find_if(one.patches.begin(), one.patches.end(),
                         [&](auto const& p) { return p.box == patch.box; })
            - one.patches.begin());
        ASSERT_LT(j, size(one)) << "patch " << patch.id;
        auto& f  = field(s.ld.data[i], name);
        auto& f1 = field(one.data[j], name);
        core::for_each_point(ghost_box(f, patch.box), [&](auto const& p) {
            auto const x = value(f, patch.box, p), y = value(f1, patch.box, p);
            EXPECT_TRUE(std::bit_cast<std::uint64_t>(x) == std::bit_cast<std::uint64_t>(y)
                        or (std::isnan(x) and std::isnan(y)))
                << name << " patch " << patch.id << " " << p << " " << x << " vs " << y;
        });
    }
}

INSTANTIATE_TEST_SUITE_P(Mpi, MpiGhostFill,
                         ::testing::Combine(::testing::Values("Bx", "rho", "Ez", "alpha_density"),
                                            ::testing::Bool()));


class MpiBorders : public ::testing::TestWithParam<bool> // periodic
{
};

TEST_P(MpiBorders, sumCountsEveryPatchImageCoveringAPoint)
{
    World s{GetParam()};
    for_each_patch(s.ld, [&](auto const& patch, auto& data) {
        core::for_each_point(ghost_box(data.rho, patch.box),
                             [&](auto const& p) { value(data.rho, patch.box, p) = 1; });
    });

    mpi::copy(sum_borders, s.n, {"rho"}, s.ld);

    for_each_patch(s.ld, [&](auto const& patch, auto& data) {
        auto const imgs = images(s.h, [&](auto const& p) { return ghost_box(data.rho, p.box); });
        core::for_each_point(ghost_box(data.rho, patch.box), [&](auto const& p) {
            double count = 0;
            for (auto const& [_, b] : imgs)
                count += isIn(p, b);
            EXPECT_EQ(value(data.rho, patch.box, p), count) << "patch " << patch.id << " " << p;
        });
    });
}

TEST_P(MpiBorders, maxIsTheLargestOfEveryPatchImageCoveringAPoint)
{
    World s{GetParam()};
    for_each_patch(s.ld, [&](auto const& patch, auto& data) {
        core::for_each_point(ghost_box(data.Bx, patch.box),
                             [&](auto const& p) { value(data.Bx, patch.box, p) = patch.id + 1.; });
    });

    mpi::copy(max_borders, s.n, {"Bx"}, s.ld);

    for_each_patch(s.ld, [&](auto const& patch, auto& data) {
        auto const imgs = images(s.h, [&](auto const& p) { return ghost_box(data.Bx, p.box); });
        core::for_each_point(ghost_box(data.Bx, patch.box), [&](auto const& p) {
            double max = 0;
            for (auto const& [img, b] : imgs)
                if (isIn(p, b))
                    max = std::max(max, img.id + 1.);
            EXPECT_EQ(value(data.Bx, patch.box, p), max) << "patch " << patch.id << " " << p;
        });
    });
}

INSTANTIATE_TEST_SUITE_P(Mpi, MpiBorders, ::testing::Bool());


class MpiLeaving : public ::testing::TestWithParam<bool> // periodic
{
};

TEST_P(MpiLeaving, particlesArriveWhereTheirCellIsOwned)
{
    bool const periodic = GetParam();
    World s{periodic};

    std::size_t n_left = 0, n_inside = 0;
    auto const level_cells = level_domain(s.h, 0);
    for_each_patch(s.ld, [&](auto const& patch, auto& data) {
        core::for_each_point(ghost_box(data.patch_ghost, patch.box), [&](auto const& c) {
            if (!isIn(c, patch.box))
            {
                data.patch_ghost.items.push_back({{c[0], c[1]}, {c[0], c[1]}, patch.id});
                ++n_left;
                n_inside += isIn(c, level_cells);
            }
        });
    });

    mpi::copy(gather_leaving, s.n, std::vector<ResourcePair>{{"patch_ghost", "domain"}}, s.ld);

    auto const shape = level_cells.shape();
    auto const wrap  = [&](std::array<int, 2> c) {
        for (std::size_t i = 0; i < 2; ++i)
            if (periodic)
                c[i] = mod(c[i], shape[i]);
        return c;
    };

    std::size_t n_arrived = 0;
    for_each_patch(s.ld, [&](auto const& patch, auto& data) {
        for (auto const& item : data.domain.items)
        {
            ++n_arrived;
            EXPECT_TRUE(isIn(Point2{item.cell}, patch.box)) << patch.id;
            EXPECT_EQ(item.cell, wrap(item.origin));
        }
    });

    auto const arrived = sum_over_ranks(n_arrived);
    EXPECT_GT(arrived, 0u);
    EXPECT_EQ(arrived, sum_over_ranks(periodic ? n_left : n_inside));
}

INSTANTIATE_TEST_SUITE_P(Mpi, MpiLeaving, ::testing::Bool());


// the caller's accessors over ranks: the same bits as by names (a fill, an order dependent
//  border sum, particles leaving), the pieces packed and unpacked through them
namespace
{
    template<auto member>
    struct Member
    {
        auto& src(toy::PatchData& d) const { return d.*member; }
        auto& dst(toy::PatchData& d) const { return d.*member; }
    };
    struct GhostsToDomain
    {
        auto& src(toy::PatchData& d) const { return d.patch_ghost; }
        auto& dst(toy::PatchData& d) const { return d.domain; }
    };
} // namespace

TEST(MpiCopy, accessorsGiveWhatNamesGive)
{
    for (auto const periodic : {false, true})
    {
        World a{periodic}, b{periodic};
        for (auto* w : {&a, &b})
            for_each_patch(w->ld, [&](auto const& patch, auto& data) {
                core::for_each_point(domain_box(data.Bx, patch.box), [&](auto const& p) {
                    value(data.Bx, patch.box, p) = f_of(w->h, p);
                });
                core::for_each_point(ghost_box(data.rho, patch.box), [&](auto const& p) {
                    value(data.rho, patch.box, p)
                        = 0.1 + 1. / (3. + patch.id) + 1e-3 * f_of(w->h, p) / 7.;
                });
                core::for_each_point(ghost_box(data.patch_ghost, patch.box), [&](auto const& c) {
                    if (!isIn(c, patch.box))
                        data.patch_ghost.items.push_back({{c[0], c[1]}, {c[0], c[1]}, patch.id});
                });
            });
        mpi::copy(fill_ghosts, a.n, {"Bx"}, a.ld);
        mpi::copy(sum_borders, a.n, {"rho"}, a.ld);
        mpi::copy(gather_leaving, a.n, std::vector<ResourcePair>{{"patch_ghost", "domain"}}, a.ld);
        mpi::copy(fill_ghosts, b.n, same_names({"Bx"}), std::array{Member<&toy::PatchData::Bx>{}},
                  b.ld);
        mpi::copy(sum_borders, b.n, same_names({"rho"}), std::array{Member<&toy::PatchData::rho>{}},
                  b.ld);
        mpi::copy(gather_leaving, b.n, std::vector<ResourcePair>{{"patch_ghost", "domain"}},
                  std::array{GhostsToDomain{}}, b.ld);

        for (std::size_t i = 0; i < size(a.ld); ++i)
        {
            auto const& box = a.ld.patches[i].box;
            for (auto const f : {&toy::PatchData::Bx, &toy::PatchData::rho})
                core::for_each_point(ghost_box(a.ld.data[i].*f, box), [&](auto const& p) {
                    auto const x = value(a.ld.data[i].*f, box, p);
                    auto const y = value(b.ld.data[i].*f, box, p);
                    EXPECT_TRUE(std::bit_cast<std::uint64_t>(x) == std::bit_cast<std::uint64_t>(y)
                                or (std::isnan(x) and std::isnan(y)))
                        << i << " " << p;
                });
            ASSERT_EQ(a.ld.data[i].domain.items.size(), b.ld.data[i].domain.items.size()) << i;
        }
    }
}

// damn::mpi::Copy driven by ThreadPool::patch_exec's hooks (patches in reverse, each one's
//  sources written in its task, sent as patches finish, received while others run): the same
//  bits as mpi::copy
TEST(MpiCopy, inFlightGivesWhatCopyGives)
{
    auto& pool = core::ThreadPool::INSTANCE();
    pool.configure(3, 1);

    auto const init = [](auto const& h, auto const& patch, auto& data) {
        core::for_each_point(domain_box(data.Bx, patch.box),
                             [&](auto const& p) { value(data.Bx, patch.box, p) = f_of(h, p); });
        core::for_each_point(ghost_box(data.rho, patch.box), [&](auto const& p) {
            value(data.rho, patch.box, p) = 0.1 + 1. / (3. + patch.id) + 1e-3 * f_of(h, p) / 7.;
        });
        core::for_each_point(ghost_box(data.patch_ghost, patch.box), [&](auto const& c) {
            if (!isIn(c, patch.box))
                data.patch_ghost.items.push_back({{c[0], c[1]}, {c[0], c[1]}, patch.id});
        });
    };

    for (auto const periodic : {false, true})
    {
        World a{periodic}, b{periodic};
        for_each_patch(a.ld, [&](auto const& patch, auto& data) { init(a.h, patch, data); });
        mpi::copy(fill_ghosts, a.n, {"Bx"}, a.ld);
        mpi::copy(sum_borders, a.n, {"rho"}, a.ld);
        mpi::copy(gather_leaving, a.n, std::vector<ResourcePair>{{"patch_ghost", "domain"}}, a.ld);

        std::vector<std::size_t> order(size(b.ld));
        for (std::size_t k = 0; k < order.size(); ++k)
            order[k] = order.size() - 1 - k;

        auto const in_flight = [&](auto const op, std::vector<ResourcePair> const& pairs,
                                   bool const first) {
            mpi::Copy c{op, b.n, pairs, by_name, b.ld};
            if (first) // feeders first, each patch once
            {
                std::vector<std::size_t> sorted{c.remote_first().begin(), c.remote_first().end()};
                EXPECT_EQ(sorted.size(), size(b.ld));
                std::sort(sorted.begin(), sorted.end());
                for (std::size_t k = 0; k < sorted.size(); ++k)
                    EXPECT_EQ(sorted[k], k);
            }
            pool.patch_exec(
                size(b.ld), [](std::size_t const i) { return i; },
                [&](std::size_t const i) {
                    if (first)
                        init(b.h, b.ld.patches[i], b.ld.data[i]);
                },
                core::ThreadPool::PatchHooks{.order   = order,
                                             .on_done = [&](std::size_t const i) { c.done(i); },
                                             .idle    = [&] { c.poll(); }});
            c.finish();
        };
        in_flight(fill_ghosts, same_names({"Bx"}), true);
        in_flight(sum_borders, same_names({"rho"}), false);
        in_flight(gather_leaving, {{"patch_ghost", "domain"}}, false);

        for (std::size_t i = 0; i < size(a.ld); ++i)
        {
            auto const& box = a.ld.patches[i].box;
            for (auto const f : {&toy::PatchData::Bx, &toy::PatchData::rho})
                core::for_each_point(ghost_box(a.ld.data[i].*f, box), [&](auto const& p) {
                    auto const x = value(a.ld.data[i].*f, box, p);
                    auto const y = value(b.ld.data[i].*f, box, p);
                    EXPECT_TRUE(std::bit_cast<std::uint64_t>(x) == std::bit_cast<std::uint64_t>(y)
                                or (std::isnan(x) and std::isnan(y)))
                        << i << " " << p;
                });
            ASSERT_EQ(a.ld.data[i].domain.items.size(), b.ld.data[i].domain.items.size()) << i;
        }
    }
    pool.configure(1, 1);
}

TEST(MpiCopy, oneCopyInFlight)
{
    World s{true};
    mpi::Copy c{sum_borders, s.n, same_names({"rho"}), by_name, s.ld};
    EXPECT_THROW((mpi::Copy{fill_ghosts, s.n, same_names({"Bx"}), by_name, s.ld}),
                 std::runtime_error);
    c.finish();
}

TEST(MpiCopy, repeatedCopiesReuseThePlanAndBuffers)
{
    World s{true};
    for (int i = 0; i < 3; ++i)
        mpi::copy(sum_borders, s.n, {"rho"}, s.ld);
    EXPECT_EQ(s.n.plans.size(), 1u);
}


#if DAMN_HAVE_GPU
// fields in unified memory: the device copies and their messages give the host's bits
namespace
{
    // values in C order over the ghost box
    template<bool device>
    struct CField
    {
        bool static constexpr on_device = device;
        std::array<bool, 2> primal{};
        int gw = 2;
        std::conditional_t<device, core::gpu::Vec_t<double>, std::vector<double>> v{};
    };
    template<bool device>
    Box2 domain_box(CField<device> const& f, Box2 cells)
    {
        for (std::size_t i = 0; i < 2; ++i)
            cells.upper[i] += f.primal[i];
        return cells;
    }
    template<bool device>
    Box2 ghost_box(CField<device> const& f, Box2 const& cells)
    {
        return core::grow(domain_box(f, cells), f.gw);
    }
    template<bool device>
    core::BoxSpan<double, 2> box_span(CField<device>& f, Box2 const& cells)
    {
        return {f.v.data(), ghost_box(f, cells)};
    }
    template<bool device>
    core::BoxSpan<double const, 2> box_span(CField<device> const& f, Box2 const& cells)
    {
        return {f.v.data(), ghost_box(f, cells)};
    }
    template<bool device>
    double& value(CField<device>& f, Box2 const& cells, Point2 const& p)
    {
        return box_span(f, cells)(p);
    }
    template<bool device>
    double const& value(CField<device> const& f, Box2 const& cells, Point2 const& p)
    {
        return box_span(f, cells)(p);
    }

    struct DevicePatchData // per centering, a field each side
    {
        std::array<CField<false>, 2> host;
        std::array<CField<true>, 2> device;
    };
    // the plans' geometry (by name): either side, the same boxes
    void for_each_resource(DevicePatchData& d, auto&& fn)
    {
        fn("Bx", d.host[0]);
        fn("rho", d.host[1]);
    }
    template<bool device, std::size_t i>
    struct Side
    {
        auto& src(DevicePatchData& d) const
        {
            if constexpr (device)
                return d.device[i];
            else
                return d.host[i];
        }
        auto& dst(DevicePatchData& d) const { return src(d); }
    };
} // namespace

TEST(MpiDeviceCopy, theDeviceGivesTheHostsBits)
{
    static_assert(detail::DeviceSource<CField<true>, 2>
                  and detail::DeviceDestination<CField<true>, 2>);
    static_assert(!detail::OnDevice<CField<false>>);
    for (auto const periodic : {false, true})
    {
        auto const h    = make_hierarchy({periodic, periodic}, mpi_size());
        auto const make = [](Patch<2> const& patch) {
            DevicePatchData d;
            for (std::size_t i = 0; i < 2; ++i)
            {
                d.host[i].primal = d.device[i].primal = {i == 0, true}; // Bx like, rho like
                d.host[i].v.assign(ghost_box(d.host[i], patch.box).size(), 0.);
                d.device[i].v.assign(d.host[i].v.size(), 0.);
            }
            return d;
        };
        auto ld = allocate(h.levels[0], mpi_rank(), make);
        // values whose sums depend on the order they are added in
        for_each_patch(ld, [&](auto const& patch, auto& d) {
            for (std::size_t i = 0; i < 2; ++i)
                core::for_each_point(ghost_box(d.host[i], patch.box), [&](auto const& p) {
                    auto const v = 0.1 + 1. / (3. + patch.id) + 1e-3 * f_of(h, p) / 7.;
                    value(d.host[i], patch.box, p)   = v;
                    value(d.device[i], patch.box, p) = v;
                });
        });

        auto run = [&]<bool device>() { // plans of their own: a plan keeps its device table
            auto n = make_neighbours(h, 0, mpi_rank(), mpi::topology(), reach);
            mpi::copy(fill_ghosts, n, same_names({"Bx"}), std::array{Side<device, 0>{}}, ld);
            mpi::copy(sum_borders, n, same_names({"rho"}), std::array{Side<device, 1>{}}, ld);
            mpi::copy(max_borders, n, same_names({"Bx"}), std::array{Side<device, 0>{}}, ld);
            for (auto const& [_, plan] : n.plans)
                EXPECT_EQ(static_cast<amr::detail::DeviceTable<2>*>(plan.device.get())->usable,
                          device);
        };
        run.template operator()<true>();
        run.template operator()<false>();

        std::size_t compared = 0;
        for_each_patch(ld, [&](auto const& patch, auto& d) {
            for (std::size_t i = 0; i < 2; ++i)
                core::for_each_point(ghost_box(d.host[i], patch.box), [&](auto const& p) {
                    auto const x = value(d.host[i], patch.box, p);
                    auto const y = value(d.device[i], patch.box, p);
                    EXPECT_EQ(std::bit_cast<std::uint64_t>(x), std::bit_cast<std::uint64_t>(y))
                        << periodic << " " << i << " patch " << patch.id << " " << p;
                    ++compared;
                });
        });
        EXPECT_GT(sum_over_ranks(compared), 0u);
    }
}

namespace
{
    // on the device, its values reached point by point only (no box_span)
    struct DevicePointField
    {
        bool static constexpr on_device = true;
        CField<true> f;
    };
    [[maybe_unused]] Box2 domain_box(DevicePointField const& f, Box2 const& cells)
    {
        return domain_box(f.f, cells);
    }
    [[maybe_unused]] Box2 ghost_box(DevicePointField const& f, Box2 const& cells)
    {
        return ghost_box(f.f, cells);
    }
    double& value(DevicePointField& f, Box2 const& cells, Point2 const& p)
    {
        return value(f.f, cells, p);
    }
    double const& value(DevicePointField const& f, Box2 const& cells, Point2 const& p)
    {
        return value(f.f, cells, p);
    }

    struct PointPatchData
    {
        DevicePointField Bx;
    };
    void for_each_resource(PointPatchData& d, auto&& fn)
    {
        fn("Bx", d.Bx);
    }
} // namespace

// the host path would read device values on the host
TEST(MpiDeviceCopy, aDeviceResourceTheDeviceCannotTakeThrows)
{
    auto const h    = make_hierarchy({true, true}); // every patch on this rank
    auto const make = [](Patch<2> const& patch) {
        PointPatchData d;
        d.Bx.f.primal = {true, false};
        d.Bx.f.v.assign(ghost_box(d.Bx, patch.box).size(), 0.);
        return d;
    };
    auto ld = allocate(h.levels[0], 0, make);
    auto n  = neighbours_of(h);
    EXPECT_THROW(copy(fill_ghosts, n, {"Bx"}, ld), std::runtime_error);
}
#if DAMN_HAVE_THRUST
// items in unified memory, grouped by cell: the device's gather over ranks gives the host's items
namespace
{
    struct DeviceParticles
    {
        using value_type                = toy::Item;
        bool static constexpr on_device = true;
        int gw                          = 1;
        Box2 cells{}; // its ghost cells: starts and sizes over them
        bool grouped = true;
        core::gpu::Vec_t<std::uint32_t> starts, sizes;
        core::gpu::Vec_t<int> cx, cy, ox, oy; // cell, origin
        core::gpu::Vec_t<std::uint32_t> from;

        std::size_t size() const { return cx.size(); }
    };
    [[maybe_unused]] Box2 domain_box(DeviceParticles const&, Box2 const& cells)
    {
        return cells;
    }
    Box2 ghost_box(DeviceParticles const& p, Box2 const& cells)
    {
        return core::grow(cells, p.gw);
    }
    [[maybe_unused]] void for_each_item(DeviceParticles const& p, auto&& fn)
    {
        for (std::size_t i = 0; i < p.size(); ++i)
            fn(toy::Item{{p.cx[i], p.cy[i]}, {p.ox[i], p.oy[i]}, p.from[i]});
    }
    [[maybe_unused]] void append(DeviceParticles&, toy::Item, Point2 const&)
    {
        throw std::logic_error("the host path for device items");
    }

    ItemColumns layout_of(DeviceParticles const&)
    {
        return {{4, 4, 4}, 3};
    }

    // the test's own grouping, on the host: counting sort by cell
    void group(DeviceParticles& p)
    {
        std::vector<toy::Item> items;
        for_each_item(p, [&](auto const& item) { items.push_back(item); });
        p.starts.assign(p.cells.size(), 0);
        p.sizes.assign(p.cells.size(), 0);
        auto const flat = [&](auto const& c) {
            return static_cast<std::size_t>((c[0] - p.cells.lower[0]) * (p.cells.shape()[1])
                                            + (c[1] - p.cells.lower[1]));
        };
        for (auto const& item : items)
            ++p.sizes[flat(item.cell)];
        for (std::size_t c = 1; c < p.cells.size(); ++c)
            p.starts[c] = p.starts[c - 1] + p.sizes[c - 1];
        std::vector<std::uint32_t> next(p.starts.begin(), p.starts.end());
        for (auto const& item : items)
        {
            auto const i = next[flat(item.cell)]++;
            p.cx[i]      = item.cell[0];
            p.cy[i]      = item.cell[1];
            p.ox[i]      = item.origin[0];
            p.oy[i]      = item.origin[1];
            p.from[i]    = item.from_patch;
        }
        p.grouped = true;
    }

    DeviceItems<2> device_items(DeviceParticles const& p)
    {
        if (!p.grouped)
            throw std::runtime_error("device items not grouped by cell");
        DeviceItems<2> items{p.cells, p.starts.data(), p.sizes.data(), {}, layout_of(p)};
        items.columns[0] = reinterpret_cast<std::byte const*>(p.ox.data());
        items.columns[1] = reinterpret_cast<std::byte const*>(p.oy.data());
        items.columns[2] = reinterpret_cast<std::byte const*>(p.from.data());
        return items;
    }

    DeviceItemsOut<2> device_items_appended(DeviceParticles& p, std::size_t const n)
    {
        auto const at = p.size();
        for (auto* v : {&p.cx, &p.cy, &p.ox, &p.oy})
            v->resize(at + n);
        p.from.resize(at + n);
        p.grouped = false;
        DeviceItemsOut<2> out{{}, layout_of(p), {p.cx.data() + at, p.cy.data() + at}};
        out.columns[0] = reinterpret_cast<std::byte*>(p.ox.data() + at);
        out.columns[1] = reinterpret_cast<std::byte*>(p.oy.data() + at);
        out.columns[2] = reinterpret_cast<std::byte*>(p.from.data() + at);
        return out;
    }

    void items_appended(DeviceParticles& p)
    {
        group(p);
    }

    struct ItemsPatchData
    {
        toy::Particles host_ghost, host_domain;
        DeviceParticles device_ghost, device_domain;
    };
    // the plans' geometry (by name): either side, the same boxes
    void for_each_resource(ItemsPatchData& d, auto&& fn)
    {
        fn("patch_ghost", d.host_ghost);
        fn("domain", d.host_domain);
    }
    template<bool device>
    struct Leaving
    {
        auto& src(ItemsPatchData& d) const
        {
            if constexpr (device)
                return d.device_ghost;
            else
                return d.host_ghost;
        }
        auto& dst(ItemsPatchData& d) const
        {
            if constexpr (device)
                return d.device_domain;
            else
                return d.host_domain;
        }
    };

    std::vector<std::array<int, 5>> sorted_items(auto const& particles)
    {
        std::vector<std::array<int, 5>> items;
        for_each_item(particles, [&](auto const& i) {
            items.push_back(
                {i.cell[0], i.cell[1], i.origin[0], i.origin[1], static_cast<int>(i.from_patch)});
        });
        std::sort(items.begin(), items.end());
        return items;
    }
} // namespace

TEST(MpiDeviceCopy, leavingItemsOnTheDeviceAreTheHosts)
{
    static_assert(DeviceItemSource<DeviceParticles, 2>
                  and DeviceItemDestination<DeviceParticles, 2>);
    for (auto const periodic : {false, true})
    {
        auto const h    = make_hierarchy({periodic, periodic}, mpi_size());
        auto const make = [](Patch<2> const& patch) {
            ItemsPatchData d;
            for (auto* p : {&d.device_ghost, &d.device_domain})
            {
                p->cells = ghost_box(*p, patch.box);
                group(*p);
            }
            return d;
        };
        auto ld = allocate(h.levels[0], mpi_rank(), make);
        // per ghost cell outside the patch, as many items as the cell's row, some none
        for_each_patch(ld, [&](auto const& patch, auto& d) {
            core::for_each_point(ghost_box(d.host_ghost, patch.box), [&](auto const& c) {
                if (isIn(c, patch.box))
                    return;
                for (int k = 0; k < mod(c[1], 3); ++k)
                {
                    toy::Item const item{{c[0], c[1]}, {c[0] + 100 * k, c[1]}, patch.id};
                    d.host_ghost.items.push_back(item);
                    d.device_ghost.cx.push_back(item.cell[0]);
                    d.device_ghost.cy.push_back(item.cell[1]);
                    d.device_ghost.ox.push_back(item.origin[0]);
                    d.device_ghost.oy.push_back(item.origin[1]);
                    d.device_ghost.from.push_back(item.from_patch);
                }
            });
            group(d.device_ghost);
        });

        auto run = [&]<bool device>() {
            auto n = make_neighbours(h, 0, mpi_rank(), mpi::topology(), reach);
            mpi::copy(gather_leaving, n, std::vector<ResourcePair>{{"patch_ghost", "domain"}},
                      std::array{Leaving<device>{}}, ld);
            for (auto const& [_, plan] : n.plans)
                EXPECT_EQ(static_cast<amr::detail::ItemTable<2>*>(plan.device.get())->usable,
                          device);
        };
        run.template operator()<true>();
        run.template operator()<false>();

        std::size_t arrived = 0;
        for_each_patch(ld, [&](auto const& patch, auto& d) {
            auto const host = sorted_items(d.host_domain);
            EXPECT_EQ(sorted_items(d.device_domain), host) << periodic << " patch " << patch.id;
            arrived += host.size();
        });
        EXPECT_GT(sum_over_ranks(arrived), 0u);
    }
}
#endif // DAMN_HAVE_THRUST
#endif // DAMN_HAVE_GPU

} // namespace damn::amr


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    int provided = 0; // MPI from the main thread only, ThreadPool's threads beside it
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    auto const result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
