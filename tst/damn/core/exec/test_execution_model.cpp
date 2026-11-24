// damn::core::ExecutionModel: defaults, leases under the cap, given back, from many threads at once

#include "damn/core/exec/execution_model.hpp"
#include "damn/core/exec/leased_pool.hpp"

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <mutex>
#include <set>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace damn::core
{

TEST(ExecutionModel, theDefaultIsTheMainThreadWithoutMPI)
{
    ExecutionModel m;
    EXPECT_EQ(m.mpi_ranks, 0);
    EXPECT_EQ(m.usable_threads, 1u);
    EXPECT_EQ(m.threads_used, 1u);
    EXPECT_EQ(m.available(), 0u); // floor(1.2) = 1: the main thread
    EXPECT_EQ(m.use_threads(4).size(), 0u);
    EXPECT_NO_THROW(m.check());
}

TEST(ExecutionModel, modes)
{
    ExecutionModel const serial, threads{.usable_threads = 4}, ranks{.mpi_ranks = 4},
        both{.mpi_ranks = 4, .usable_threads = 4};
    EXPECT_FALSE(serial.mpi() or serial.threaded() or serial.mpi_only());
    EXPECT_TRUE(threads.threaded() and !threads.mpi() and !threads.mpi_only());
    EXPECT_TRUE(ranks.mpi() and ranks.mpi_only() and !ranks.threaded());
    EXPECT_TRUE(both.mpi() and both.threaded() and !both.mpi_only());
}

TEST(ExecutionModel, leasesWhatIsFreeUnderTheCap)
{
    ExecutionModel m{.usable_threads = 8}; // 9 schedulable, 8 free
    auto a = m.use_threads(4);
    EXPECT_EQ(a.size(), 4u);
    auto b = m.use_threads(6); // 4 left
    EXPECT_EQ(b.size(), 4u);
    EXPECT_EQ(m.threads_used, 9u);
    EXPECT_EQ(m.use_threads(1).size(), 0u);

    a.release();
    EXPECT_EQ(m.threads_used, 5u);
    {
        auto c = m.use_threads(10);
        EXPECT_EQ(c.size(), 4u);
    }
    EXPECT_EQ(m.threads_used, 5u); // c given back

    m.max_scheduled = 1.5; // 12
    EXPECT_EQ(m.use_threads(10).size(), 7u);
}

TEST(ExecutionModel, aMovedLeaseIsGivenBackOnce)
{
    ExecutionModel m{.usable_threads = 4};
    {
        auto a = m.use_threads(2);
        auto b = std::move(a);
        EXPECT_EQ(a.size(), 0u); // NOLINT(bugprone-use-after-move): moved from, empty
        EXPECT_EQ(b.size(), 2u);
        ThreadLease c;
        c = std::move(b);
        EXPECT_EQ(m.threads_used, 3u);
    }
    EXPECT_EQ(m.threads_used, 1u);
}

TEST(ExecutionModel, concurrentLeasesNeverPassTheCap)
{
    ExecutionModel m{.usable_threads = 16}; // 19 schedulable
    std::atomic<std::size_t> most{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t)
        threads.emplace_back([&] {
            for (int i = 0; i < 2000; ++i)
            {
                auto const lease = m.use_threads(3);
                auto const used  = m.threads_used.load();
                for (auto seen = most.load();
                     used > seen and !most.compare_exchange_weak(seen, used);)
                {
                }
            }
        });
    for (auto& t : threads)
        t.join();
    EXPECT_LE(most, m.schedulable());
    EXPECT_EQ(m.threads_used, 1u);
}

TEST(LeasedPool, runsSideBySideOnItsLease)
{
    ExecutionModel m{.usable_threads = 4}; // 4 schedulable: the main thread, a pool of 3
    LeasedPool pool{m};
    EXPECT_EQ(pool.threads(), 3u);
    std::mutex lock;
    std::set<std::thread::id> ids;
    std::vector<int> ran(8, 0);
    std::size_t most = 0;
    pool.run(ran.size(), [&](std::size_t const i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::lock_guard const g{lock};
        ++ran[i];
        ids.insert(std::this_thread::get_id());
        most = std::max<std::size_t>(most, m.threads_used);
    });
    EXPECT_EQ(ran, std::vector<int>(8, 1));
    EXPECT_EQ(most, 4u);
    EXPECT_EQ(ids.size(), 3u);
    EXPECT_FALSE(ids.contains(std::this_thread::get_id()));
    EXPECT_EQ(m.threads_used, 1u);

    auto const taken = m.use_threads(3); // nothing left: on the calling thread
    ids.clear();
    pool.run(4, [&](std::size_t) { ids.insert(std::this_thread::get_id()); });
    EXPECT_EQ(ids, std::set{std::this_thread::get_id()});
}

TEST(LeasedPool, theMainThreadOnly)
{
    ExecutionModel m;
    LeasedPool pool{m};
    EXPECT_EQ(pool.threads(), 0u);
    int n = 0;
    pool.run(5, [&](std::size_t) { ++n; });
    EXPECT_EQ(n, 5);
}

TEST(ExecutionModel, nothingToRunOnThrows)
{
    EXPECT_THROW(ExecutionModel{.mpi_ranks = -1}.check(), std::runtime_error);
    EXPECT_THROW(ExecutionModel{.usable_threads = 0}.check(), std::runtime_error);
    EXPECT_THROW(ExecutionModel{.max_scheduled = 0}.check(), std::runtime_error);
}

} // namespace damn::core


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
