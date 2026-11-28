#ifndef DAMN_CORE_EXEC_LEASED_POOL_HPP
#define DAMN_CORE_EXEC_LEASED_POOL_HPP

// LeasedPool: threads that run work side by side as an ExecutionModel lets them. The pool holds
// the most a lease could give (schedulable - 1: the main thread is in use) idle between runs;
// run(n, fn) leases up to n of them for the call, fn(i) for every i in [0, n) on them, on the
// calling thread if none are free

#include "BS_thread_pool.hpp"
#include "damn/core/def.hpp"
#include "damn/core/exec/execution_model.hpp"

#include <cstddef>
#include <memory>

namespace damn::core
{
struct LeasedPool
{
    explicit LeasedPool(ExecutionModel& model);

    // fn(i) for every i in [0, n) has run on return, rethrows a task's exception
    void run(std::size_t n, auto&& fn);

    MKN_KUL_FN_SIG_RET std::size_t threads() const; // the pool's, 0 without one

private:
    ExecutionModel* model_ = nullptr;
    std::unique_ptr<::BS::thread_pool<::BS::tp::none>> pool_{};
};

} // namespace damn::core


namespace damn::core
{
inline LeasedPool::LeasedPool(ExecutionModel& model)
    : model_{&model}
{
    if (auto const most = model.schedulable(); most > 1)
        pool_ = std::make_unique<::BS::thread_pool<::BS::tp::none>>(most - 1);
}

inline std::size_t LeasedPool::threads() const
{
    return pool_ ? pool_->get_thread_count() : 0;
}

void LeasedPool::run(std::size_t const n, auto&& fn)
{
    auto lease = (pool_ and n > 1) ? model_->use_threads(n) : ThreadLease{};
    if (lease.size() == 0)
    {
        for (std::size_t i = 0; i < n; ++i)
            fn(i);
        return;
    }
    pool_
        ->submit_blocks<std::size_t>(
            0, n,
            [&](std::size_t const b, std::size_t const e) {
                for (auto i = b; i < e; ++i)
                    fn(i);
            },
            lease.size())
        .get();
}

} // namespace damn::core

#endif // DAMN_CORE_EXEC_LEASED_POOL_HPP
