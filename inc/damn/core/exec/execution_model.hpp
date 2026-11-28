#ifndef DAMN_CORE_EXEC_EXECUTION_MODEL_HPP
#define DAMN_CORE_EXEC_EXECUTION_MODEL_HPP

// what one rank runs on, and the threads in use from it: doc/core/exec/execution_model.md
// pools size themselves from a lease, so pools in different contexts share one budget

#include "damn/core/def.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace damn::core
{
struct ExecutionModel;

// threads leased from an ExecutionModel, given back on destruction
struct ThreadLease
{
    ThreadLease() = default;
    ThreadLease(ExecutionModel& model, std::size_t threads);
    ThreadLease(ThreadLease const&)            = delete;
    ThreadLease& operator=(ThreadLease const&) = delete;
    ThreadLease(ThreadLease&& that) noexcept;
    ThreadLease& operator=(ThreadLease&& that) noexcept;
    ~ThreadLease();

    MKN_KUL_FN_SIG_RET std::size_t size() const; // threads leased, 0 if none were free
    void release();                              // gives them back now

private:
    ExecutionModel* model_ = nullptr;
    std::size_t threads_   = 0;
};

struct ExecutionModel
{
    int mpi_ranks                         = 0;
    std::size_t usable_threads            = 1;
    double max_scheduled                  = 1.2;
    std::atomic<std::size_t> threads_used = 1; // the main thread

    MKN_KUL_FN_SIG_RET bool mpi() const;                // mpi_ranks > 0
    MKN_KUL_FN_SIG_RET bool threaded() const;           // usable_threads > 1
    MKN_KUL_FN_SIG_RET bool mpi_only() const;           // mpi() and not threaded()
    MKN_KUL_FN_SIG_RET std::size_t schedulable() const; // the most threads_used may be
    MKN_KUL_FN_SIG_RET std::size_t available() const;   // free now
    void check() const;                                 // throws what is not runnable

    // up to n more threads, what is free when called
    MKN_KUL_FN_SIG_RET ThreadLease use_threads(std::size_t n);
};

} // namespace damn::core


namespace damn::core
{
inline bool ExecutionModel::mpi() const
{
    return mpi_ranks > 0;
}

inline bool ExecutionModel::threaded() const
{
    return usable_threads > 1;
}

inline bool ExecutionModel::mpi_only() const
{
    return mpi() and !threaded();
}

inline std::size_t ExecutionModel::schedulable() const
{
    return std::max<std::size_t>(1, static_cast<std::size_t>(std::floor(
                                        max_scheduled * static_cast<double>(usable_threads))));
}

inline std::size_t ExecutionModel::available() const
{
    auto const used = threads_used.load();
    auto const most = schedulable();
    return used < most ? most - used : 0;
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
    if (!(max_scheduled > 0))
        fail("max_scheduled must be > 0");
}

inline ThreadLease ExecutionModel::use_threads(std::size_t const n)
{
    auto used = threads_used.load();
    std::size_t granted;
    do
    {
        auto const most = schedulable();
        granted         = std::min(n, used < most ? most - used : 0);
    } while (granted > 0 and !threads_used.compare_exchange_weak(used, used + granted));
    return ThreadLease{*this, granted};
}


inline ThreadLease::ThreadLease(ExecutionModel& model, std::size_t const threads)
    : model_{&model}
    , threads_{threads}
{
}

inline ThreadLease::ThreadLease(ThreadLease&& that) noexcept
    : model_{std::exchange(that.model_, nullptr)}
    , threads_{std::exchange(that.threads_, 0)}
{
}

inline ThreadLease& ThreadLease::operator=(ThreadLease&& that) noexcept
{
    if (this != &that)
    {
        release();
        model_   = std::exchange(that.model_, nullptr);
        threads_ = std::exchange(that.threads_, 0);
    }
    return *this;
}

inline ThreadLease::~ThreadLease()
{
    release();
}

inline std::size_t ThreadLease::size() const
{
    return threads_;
}

inline void ThreadLease::release()
{
    if (model_ and threads_)
        model_->threads_used -= threads_;
    threads_ = 0;
}

} // namespace damn::core

#endif // DAMN_CORE_EXEC_EXECUTION_MODEL_HPP
