#ifndef DAMN_CORE_EXEC_EXECUTION_MODEL_HPP
#define DAMN_CORE_EXEC_EXECUTION_MODEL_HPP

// ExecutionModel: what one rank runs on, and how ThreadPool divides it
//
//   mpi_ranks          0: MPI not initialised (one process, no messages); else the run's ranks
//   usable_threads     this rank's threads: 1 (the default) is the main thread only
//   n_pools            ThreadPool's pools: patches go to pools
//   threads_per_pool   a pool's threads: one patch's work split over them (tiles, cell ranges ...)
//   max_scheduled      threads scheduled at once (n_pools * threads_per_pool) at most this times
//                      usable_threads, at least 1: some oversubscription hides waits, more only
//                      contends. 1.2 by default; what suits depends on the work and the machine
//
// the main thread waits while the pools run, so it is not counted. damn/mpi/execution_model.hpp
// fills mpi_ranks from MPI

#include "damn/core/def.hpp"

#include <cstddef>

namespace damn::core
{
struct ExecutionModel
{
    int mpi_ranks                = 0;
    std::size_t usable_threads   = 1;
    std::size_t n_pools          = 1;
    std::size_t threads_per_pool = 1;
    double max_scheduled         = 1.2;

    NO_DISCARD std::size_t scheduled() const;   // n_pools * threads_per_pool
    NO_DISCARD std::size_t schedulable() const; // the most scheduled may be
    void check() const;                         // throws what is not runnable
};

// usable_threads in n_pools pools of usable_threads / n_pools threads (at least 1): one pool of
// every thread (a patch at a time, its work split), or a pool per thread (patches side by side)
NO_DISCARD ExecutionModel split(ExecutionModel model, std::size_t n_pools);

} // namespace damn::core


#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace damn::core
{
inline std::size_t ExecutionModel::scheduled() const
{
    return n_pools * threads_per_pool;
}

inline std::size_t ExecutionModel::schedulable() const
{
    return std::max<std::size_t>(1, static_cast<std::size_t>(std::floor(
                                        max_scheduled * static_cast<double>(usable_threads))));
}

inline void ExecutionModel::check() const
{
    auto const fail = [](std::string const& why) {
        throw std::runtime_error("damn::core::ExecutionModel: " + why);
    };
    if (mpi_ranks < 0)
        fail("mpi_ranks < 0");
    if (usable_threads == 0)
        fail("no usable threads");
    if (n_pools == 0 or threads_per_pool == 0)
        fail("pools and threads_per_pool must be > 0");
    if (!(max_scheduled > 0))
        fail("max_scheduled must be > 0");
    if (scheduled() > schedulable())
        fail(std::to_string(n_pools) + " pools of " + std::to_string(threads_per_pool)
             + " threads: more than " + std::to_string(schedulable()) + " for "
             + std::to_string(usable_threads) + " usable threads at max_scheduled "
             + std::to_string(max_scheduled));
}

inline ExecutionModel split(ExecutionModel model, std::size_t const n_pools)
{
    if (n_pools == 0)
        throw std::runtime_error("damn::core::split: no pools");
    model.n_pools          = n_pools;
    model.threads_per_pool = std::max<std::size_t>(1, model.usable_threads / n_pools);
    return model;
}

} // namespace damn::core

#endif // DAMN_CORE_EXEC_EXECUTION_MODEL_HPP
