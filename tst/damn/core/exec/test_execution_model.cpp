// damn::core::ExecutionModel: defaults, the scheduled threads' cap, split, ThreadPool::configure

#include "damn/core/exec/execution_model.hpp"
#include "damn/core/exec/thread_pool.hpp"

#include "gtest/gtest.h"

#include <cstddef>
#include <stdexcept>

namespace damn::core
{

TEST(ExecutionModel, theDefaultIsTheMainThreadWithoutMPI)
{
    ExecutionModel const m;
    EXPECT_EQ(m.mpi_ranks, 0);
    EXPECT_EQ(m.usable_threads, 1u);
    EXPECT_EQ(m.scheduled(), 1u);
    EXPECT_NO_THROW(m.check());
}

TEST(ExecutionModel, atMostMaxScheduledTimesTheUsableThreads)
{
    ExecutionModel m{.usable_threads = 8};
    EXPECT_EQ(m.schedulable(), 9u); // floor(1.2 * 8)
    m.n_pools = 3, m.threads_per_pool = 3;
    EXPECT_NO_THROW(m.check());
    m.threads_per_pool = 4; // 12
    EXPECT_THROW(m.check(), std::runtime_error);
    m.max_scheduled = 1.5;
    EXPECT_NO_THROW(m.check());

    ExecutionModel one{.n_pools = 2}; // floor(1.2) = 1
    EXPECT_THROW(one.check(), std::runtime_error);
    one.max_scheduled = 0.5; // never below 1
    one.n_pools       = 1;
    EXPECT_NO_THROW(one.check());
}

TEST(ExecutionModel, nothingToRunOnThrows)
{
    for (auto const& m : {ExecutionModel{.mpi_ranks = -1}, ExecutionModel{.usable_threads = 0},
                          ExecutionModel{.n_pools = 0}, ExecutionModel{.threads_per_pool = 0},
                          ExecutionModel{.max_scheduled = 0}})
        EXPECT_THROW(m.check(), std::runtime_error);
}

TEST(ExecutionModel, splitDividesTheUsableThreads)
{
    ExecutionModel const m{.usable_threads = 8};
    auto const one = split(m, 1), per = split(m, 8), two = split(m, 2), more = split(m, 12);
    EXPECT_EQ(one.threads_per_pool, 8u);
    EXPECT_EQ(per.threads_per_pool, 1u);
    EXPECT_EQ(two.threads_per_pool, 4u);
    EXPECT_EQ(more.threads_per_pool, 1u);
    EXPECT_NO_THROW(two.check());
    EXPECT_THROW(more.check(), std::runtime_error); // 12 > 9
    EXPECT_THROW(std::ignore = split(m, 0), std::runtime_error);
}

TEST(ExecutionModel, configuresTheThreadPool)
{
    auto& tp = ThreadPool::INSTANCE();
    tp.configure(split(ExecutionModel{.usable_threads = 4}, 2));
    EXPECT_EQ(ThreadPool::n_pools, 2u);
    EXPECT_EQ(ThreadPool::threads_per_pool, 2u);
    EXPECT_THROW(tp.configure(ExecutionModel{.n_pools = 4}), std::runtime_error);
    EXPECT_EQ(ThreadPool::n_pools, 2u); // unchanged
    tp.configure(ExecutionModel{});
}

} // namespace damn::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
