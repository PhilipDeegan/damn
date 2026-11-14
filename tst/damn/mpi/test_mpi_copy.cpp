// damn::mpi::copy: the serial copy tests' brute force expectations with the patches spread over
// the ranks (round robin, uniform_level_0). Run with any number of ranks:
//   mpirun -n 3 ./bin/test_damn_mpi/test_mpi_copy
// one rank checks the MPI path against the serial copies

#include "damn/mpi/copy.hpp"
#include "damn/mpi/topology.hpp"
#include "damn/mpi/damn_mpi.hpp"
#include "tst/damn/amr/toy.hpp"

#include "gtest/gtest.h"

#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

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


TEST(MpiCopy, repeatedCopiesReuseThePlanAndBuffers)
{
    World s{true};
    for (int i = 0; i < 3; ++i)
        mpi::copy(sum_borders, s.n, {"rho"}, s.ld);
    EXPECT_EQ(s.n.plans.size(), 1u);
}

} // namespace damn::amr


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    auto const result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
