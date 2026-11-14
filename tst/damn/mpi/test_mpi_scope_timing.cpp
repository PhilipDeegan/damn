// damn::mpi::start_scope_timing / stop_scope_timing: one phlop timer file per rank, with the
// copies' scopes. Run with any number of ranks:
//   mpirun -n 3 ./bin/test_damn_mpi/test_mpi_scope_timing

#include "damn/core/def/phlop.hpp"
#include "damn/amr/copy.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/topology.hpp"
#include "damn/mpi/copy.hpp"
#include "damn/mpi/damn_mpi.hpp"
#include "damn/mpi/scope_timing.hpp"
#include "damn/mpi/topology.hpp"
#include "tst/damn/amr/toy.hpp"

#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

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

    std::string read_file(std::filesystem::path const& path)
    {
        std::ifstream in{path, std::ios::binary};
        return {std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
    }
} // namespace


TEST(MpiScopeTiming, aFilePerRankWithTheCopies)
{
    if (!DAMN_HAVE_PHLOP)
        GTEST_SKIP() << "built without phlop";

    auto const dir = std::filesystem::path{"damn_test_scope_timing"};
    if (mpi_rank() == 0)
    {
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    mpi::start_scope_timing((dir / "rank.").string());
    EXPECT_TRUE(mpi::scope_timing_active());
    bool receives = false; // waits for another rank's message
    {
        auto const h = make_hierarchy({true, true}, mpi_size());
        auto ld      = allocate(h.levels[0], mpi_rank(), make_patch_data);
        auto n       = make_neighbours(h, 0, mpi_rank(), mpi::topology(), reach);
        receives     = farthest(n) >= Distance::Node;
        mpi::copy(fill_ghosts, n, {"Bx"}, ld);
        mpi::copy(sum_borders, n, {"rho"}, ld);
    }
    mpi::stop_scope_timing();
    EXPECT_FALSE(mpi::scope_timing_active());

    auto const file = dir / ("rank." + std::to_string(mpi_rank()) + ".txt");
    ASSERT_TRUE(std::filesystem::exists(file));
    auto const content = read_file(file);
    for (auto const* scope : {"damn::mpi::copy<fill_ghosts>", "damn::mpi::copy<sum_borders>",
                              "damn::mpi::copy::pack_send", "damn::amr::make_neighbours"})
        EXPECT_NE(content.find(scope), std::string::npos) << scope;
    EXPECT_EQ(content.find("damn::mpi::copy::wait") != std::string::npos, receives);
}

TEST(MpiScopeTiming, envFlag)
{
    ::setenv("DAMN_TEST_FLAG", "1", 1);
    EXPECT_TRUE(mpi::env_flag("DAMN_TEST_FLAG"));
    ::setenv("DAMN_TEST_FLAG", "true", 1);
    EXPECT_TRUE(mpi::env_flag("DAMN_TEST_FLAG"));
    ::setenv("DAMN_TEST_FLAG", "0", 1);
    EXPECT_FALSE(mpi::env_flag("DAMN_TEST_FLAG"));
    ::unsetenv("DAMN_TEST_FLAG");
    EXPECT_FALSE(mpi::env_flag("DAMN_TEST_FLAG"));
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
