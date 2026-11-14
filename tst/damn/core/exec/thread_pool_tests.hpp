#ifndef DAMN_TST_CORE_EXEC_THREAD_POOL_TESTS_HPP
#define DAMN_TST_CORE_EXEC_THREAD_POOL_TESTS_HPP

// ThreadPool's tests, built threaded (test_thread_pool.cpp) and synchronous
// (test_thread_pool_sync.cpp, DAMN_THREAD_POOL_SYNC): the same results either way

#include "damn/core/exec/thread_pool.hpp"

#include "gtest/gtest.h"

#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <vector>

namespace damn::core
{
namespace
{
    struct Pools
    {
        explicit Pools(std::size_t const pools = 3, std::size_t const threads = 2)
        {
            ThreadPool::INSTANCE().configure(pools, threads);
        }
        ~Pools() { ThreadPool::INSTANCE().configure(1, 1); }
    };

    // the view of patch i
    struct View
    {
        std::size_t patch;
    };
} // namespace


TEST(ThreadPool, patchExecRunsEveryPatchOnce)
{
    Pools pools;
    std::vector<std::atomic<int>> runs(17);
    ThreadPool::INSTANCE().patch_exec(
        runs.size(), [](std::size_t const i) { return View{i}; },
        [&](View const& v) { ++runs[v.patch]; });
    for (auto const& r : runs)
        EXPECT_EQ(r, 1);
}

TEST(ThreadPool, tileExecRunsEveryTileOnce)
{
    Pools pools;
    std::size_t constexpr patches = 5, tiles = 7;
    std::vector<std::atomic<int>> runs(patches * tiles);
    ThreadPool::INSTANCE().tile_exec(
        patches, [](std::size_t const i) { return View{i}; }, [](View const&) { return tiles; },
        [&](View const& v, std::size_t const t) { ++runs[v.patch * tiles + t]; });
    for (auto const& r : runs)
        EXPECT_EQ(r, 1);
}

TEST(ThreadPool, aTaskExceptionIsRethrownOnSync)
{
    Pools pools;
    auto& tp = ThreadPool::INSTANCE();
    tp.async([] { throw std::runtime_error("task"); });
    EXPECT_THROW(tp.sync(), std::runtime_error);
    EXPECT_NO_THROW(tp.sync()); // once
}

// synchronous: on the calling thread, in submission order
TEST(ThreadPool, whereTasksRun)
{
    Pools pools;
    auto const caller = std::this_thread::get_id();
    std::vector<std::size_t> order;
    std::atomic<int> on_caller = 0;
    ThreadPool::INSTANCE().patch_exec(
        9, [](std::size_t const i) { return View{i}; },
        [&](View const& v) {
            if (std::this_thread::get_id() == caller)
            {
                ++on_caller;
                order.push_back(v.patch);
            }
        });
    if constexpr (ThreadPool::synchronous)
    {
        EXPECT_EQ(on_caller, 9);
        for (std::size_t i = 0; i < order.size(); ++i)
            EXPECT_EQ(order[i], i);
    }
    else
        EXPECT_EQ(on_caller, 0); // the pools' threads
}

} // namespace damn::core

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif // DAMN_TST_CORE_EXEC_THREAD_POOL_TESTS_HPP
