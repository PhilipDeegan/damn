// bootstrap probe (see etc/bootstrap/bootstrap.py): ranks=1 profile=bootstrap mpirun=--bind-to,none
// how many threads a rank's cores take best: T threads, pinned round robin to the rank's M
// allowed cpus, taking work items from a shared counter (as a pool's threads take patches),
// timed for T = M .. 1.5 M on a fixed amount of work. Two kernels on a cache sized array per
// item: compute only (fused multiply adds, as a push or a stencil), and the same with a block
// every 4th item (~200 us, as a wait on a message). Prints cores, per kernel the best T and the
// throughput at M + 10% and M + 25% relative to M (> 1: more threads help)
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <thread>
#include <vector>

namespace
{
std::vector<int> allowed_cpus()
{
    cpu_set_t set;
    CPU_ZERO(&set);
    sched_getaffinity(0, sizeof(set), &set);
    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &set))
            cpus.push_back(c);
    return cpus;
}

void pin(int const cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

// items per second with T threads taking `items` work items
double rate(int const threads, std::vector<int> const& cpus, std::size_t const items,
            bool const stalls)
{
    std::size_t constexpr n = 32 * 1024; // doubles an item: 256 KB, about an L2
    std::atomic<std::size_t> next{0};
    std::atomic<double> sink{0};
    auto const work = [&](int const t) {
        pin(cpus[static_cast<std::size_t>(t) % cpus.size()]);
        std::vector<double> a(n, 1.0001), b(n, 0.9999);
        double acc = 0;
        for (std::size_t i; (i = next.fetch_add(1)) < items;)
        {
            for (int pass = 0; pass < 8; ++pass)
                for (std::size_t k = 0; k < n; ++k)
                    a[k] = std::fma(a[k], b[k], 1e-9);
            acc += a[i % n];
            if (stalls and i % 4 == 0)
                std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        sink.fetch_add(acc);
    };
    auto const t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back(work, t);
    for (auto& th : pool)
        th.join();
    auto const s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return static_cast<double>(items) / s;
}
} // namespace

int main()
{
    auto const cpus = allowed_cpus();
    auto const m    = static_cast<int>(cpus.size());
    if (m < 1)
        return 1;
    std::printf("cores=%d\n", m);
    for (bool const stalls : {false, true})
    {
        // the amount of work: about half a second at M threads
        std::size_t items = 64;
        while (items < (std::size_t{1} << 20) and items / rate(m, cpus, items, stalls) < 0.25)
            items *= 2;
        double const base = rate(m, cpus, items, stalls);
        int best          = m;
        double best_rate  = base;
        double at10 = base, at25 = base;
        for (int t = m + 1; t <= m + (m + 1) / 2; ++t)
        {
            auto const r = rate(t, cpus, items, stalls);
            if (r > best_rate)
            {
                best      = t;
                best_rate = r;
            }
            if (t == m + std::max(1, (m + 9) / 10))
                at10 = r;
            if (t == m + std::max(1, (m + 3) / 4))
                at25 = r;
        }
        auto const* const kind = stalls ? "stalled" : "compute";
        std::printf("%s_best=%d\n%s_plus10=%.3f\n%s_plus25=%.3f\n", kind, best, kind, at10 / base,
                    kind, at25 / base);
    }
    return 0;
}
