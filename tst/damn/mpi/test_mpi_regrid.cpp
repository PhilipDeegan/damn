// damn::mpi::regrid: the toy level (8 x 6 cells, 3 x 4 patches, along the Hilbert curve over the
// ranks) onto other decompositions, patches changing ranks. Run with any number of ranks:
//   mpirun -n 3 ./bin/test_damn_mpi/test_mpi_regrid

#include "damn/amr/hierarchy.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/partition.hpp"
#include "damn/amr/regrid.hpp"
#include "damn/mpi/damn_mpi.hpp"
#include "damn/mpi/regrid.hpp"
#include "tst/damn/amr/toy.hpp"
#include "tst/damn/amr/toy_regrid.hpp"

#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <utility>
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
        Hierarchy<2> h = make_hierarchy({true, true}, mpi_size());
        LevelData_t ld = allocate(h.levels[0], mpi_rank(), make_patch_data);

        World() { fill(h, ld); }

        void onto(Level<2> const& level, mpi::RegridLimits const limits = {})
        {
            auto const map = make_regrid_map(h.levels[0], level, mpi_rank());
            ld             = mpi::regrid(map, std::move(ld), make_patch_data, limits);
            set_level(h, level);
        }

        void expect_correct()
        {
            EXPECT_EQ(ld.patches, local_patches(h.levels[0], mpi_rank()));
            EXPECT_EQ(sum_over_ranks(check(h, ld)), 0u);
            EXPECT_EQ(sum_over_ranks(domain_items(ld)), 48u);
        }
    };

    // the ranks reversed: every patch not on the middle rank moves
    Level<2> reversed(Level<2> level)
    {
        for (auto& patch : level.patches)
            patch.rank = mpi_size() - 1 - patch.rank;
        return level;
    }

    Level<2> rechopped(Hierarchy<2> const& h, std::array<int, 2> const size, bool const reverse)
    {
        auto const boxes = chop(h.domain, size);
        auto level       = make_level(0, boxes, partition(boxes, cell_weights(boxes), mpi_size()));
        return reverse ? reversed(level) : level;
    }
} // namespace


TEST(MpiRegrid, newOwners)
{
    World w;
    w.onto(reversed(w.h.levels[0]));
    w.expect_correct();
}

TEST(MpiRegrid, smallerPatches)
{
    World w;
    w.onto(rechopped(w.h, {2, 2}, false));
    w.expect_correct();
}

TEST(MpiRegrid, smallerPatchesNewOwners)
{
    World w;
    w.onto(rechopped(w.h, {2, 2}, true));
    w.expect_correct();
}

TEST(MpiRegrid, largerPatchesNewOwners)
{
    World w;
    w.onto(rechopped(w.h, {4, 6}, true));
    w.expect_correct();
}

// every transfer its own message, and at most one in flight: receives while sending
TEST(MpiRegrid, tinyLimits)
{
    World w;
    w.onto(rechopped(w.h, {2, 2}, true), {1, 1});
    w.expect_correct();
    w.onto(rechopped(w.h, {3, 4}, false), {1, 1}); // and back
    w.expect_correct();
}

TEST(MpiRegrid, aMismatchedMapThrows)
{
    World w;
    auto const map = make_regrid_map(w.h.levels[0], reversed(w.h.levels[0]), mpi_rank());
    auto other     = make_hierarchy({true, true}, 1); // every patch on rank 0
    auto ld        = allocate(other.levels[0], mpi_rank(), make_patch_data);
    if (mpi_size() > 1 and ld.patches != map.old_local)
        EXPECT_THROW((void)mpi::regrid(map, std::move(ld), make_patch_data), std::runtime_error);
}

} // namespace damn::amr


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    auto const result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
