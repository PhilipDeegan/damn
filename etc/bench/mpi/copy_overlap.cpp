// work modes on mock fields: per patch work, then a copy between the level's patches, timed with
// the copy after the work (whole_patch) and in flight while the work runs
// (distance_first_overlaps: damn::mpi::Copy driven by ThreadPool::patch_exec's hooks)
//
//   mpirun -n 4 --bind-to none bin/bench_damn_mpi/bench_damn_mpi cells=512 patch=32 pools=3
//
// key=value arguments (defaults):
//   cells=256     domain cells per direction (2D, periodic)
//   patch=32      patch cells per direction (uniform_level_0)
//   steps=40      timed steps per mode (after 3 warm up steps), modes alternating step by step
//   work=8        sweeps over a patch's points per step (work per point, pointwise math)
//   imbalance=0   patch work spread: sweeps x (1 + imbalance x u), u in [0, 1) per patch id
//   pools=2       damn's thread pools (patch_exec: a patch per pool), threads=1 per pool
//   mode=both     both (alternating step by step, compared bit for bit), whole_patch (all the
//                 work up front, then the copy) or distance_first_overlaps alone
//   copy=sum      sum (sum_borders: deposits' border sums) or fill (fill_ghosts)
//   latency=0     simulated network (Copy::simulate_latency): a message is received latency
//   jitter=0      + jitter x u microseconds after it is first seen, u in [0, 1) per message
//                 (step, sender, receiver): the same delays in both modes; one node's shared
//                 memory otherwise delivers at once
// Rank 0 prints per mode the median / min step (max over ranks) and whole_patch's share of
// work and copy; at the end both modes' fields are compared bit for bit (they must agree).

#include "damn/core/exec/thread_pool.hpp"
#include "damn/mpi/copy.hpp"
#include "damn/mpi/damn_mpi.hpp"
#include "damn/mpi/topology.hpp"
#include "tst/damn/amr/toy.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace damn;
using namespace damn::amr;
using Clock = std::chrono::steady_clock;

struct Args
{
    int cells = 256, patch = 32, steps = 40, work = 8;
    double imbalance = 0, latency = 0, jitter = 0;
    std::size_t pools = 2, threads = 1;
    std::string copy = "sum", mode = "both";
};

Args parse(int const argc, char** argv)
{
    std::map<std::string, std::string> kv;
    for (int i = 1; i < argc; ++i)
    {
        std::string const a = argv[i];
        auto const eq       = a.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error("copy_overlap: key=value arguments, not " + a);
        kv[a.substr(0, eq)] = a.substr(eq + 1);
    }
    Args args;
    auto const take = [&](std::string const& key, auto& value) {
        auto const it = kv.find(key);
        if (it == kv.end())
            return;
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(value)>, std::string>)
            value = it->second;
        else if constexpr (std::is_floating_point_v<std::remove_cvref_t<decltype(value)>>)
            value = std::stod(it->second);
        else
            value = static_cast<std::remove_cvref_t<decltype(value)>>(std::stoll(it->second));
        kv.erase(it);
    };
    take("cells", args.cells);
    take("patch", args.patch);
    take("steps", args.steps);
    take("work", args.work);
    take("imbalance", args.imbalance);
    take("pools", args.pools);
    take("threads", args.threads);
    take("copy", args.copy);
    take("mode", args.mode);
    take("latency", args.latency);
    take("jitter", args.jitter);
    if (!kv.empty())
        throw std::runtime_error("copy_overlap: unknown argument " + kv.begin()->first);
    if (args.mode != "both" and args.mode != "whole_patch"
        and args.mode != "distance_first_overlaps")
        throw std::runtime_error(
            "copy_overlap: mode=both, mode=whole_patch or mode=distance_first_overlaps");
    if (args.copy != "sum" and args.copy != "fill")
        throw std::runtime_error("copy_overlap: copy=sum or copy=fill");
    return args;
}

int rank_of()
{
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

int size_of()
{
    int s = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    return s;
}

double max_over_ranks(double const local)
{
    double global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global;
}

struct World
{
    explicit World(Args const& args)
        : h{[&] {
            Hierarchy<2> hier;
            hier.domain   = toy::Box2{std::array{0, 0}, std::array{args.cells - 1, args.cells - 1}};
            hier.dx       = {.1, .1};
            hier.periodic = {true, true};
            set_level(hier, uniform_level_0(hier, {args.patch, args.patch}, size_of()));
            return hier;
        }()}
        , ld{allocate(h.levels[0], rank_of(), make_patch_data)}
        , n{make_neighbours(h, 0, rank_of(), mpi::topology(), reach)}
    {
    }

    Hierarchy<2> h;
    LevelData_t ld;
    Neighbours<2> n;
};

// a patch's work for a step: its field (rho or Bx) set from the patch and step, then sweeps of
//  pointwise math over every point (ghosts too: the sums read them)
void work_on(toy::Field& f, Patch<2> const& patch, int const step, int const sweeps)
{
    auto const box = ghost_box(f, patch.box);
    std::size_t k  = 0;
    core::for_each_point(box, [&](auto const& p) {
        f.v[k++] = 1. + 1e-3 * patch.id + 1e-4 * step + 1e-6 * (p[0] + 3 * p[1]);
    });
    for (int s = 0; s < sweeps; ++s)
        for (auto& v : f.v)
            v = 0.999 * v + 1e-3 * std::sin(v);
}

int sweeps_of(Args const& args, Patch<2> const& patch)
{
    auto const u = static_cast<double>((patch.id * 2654435761u) % 1000u) / 1000.;
    return static_cast<int>(std::lround(args.work * (1. + args.imbalance * u)));
}

template<typename Op>
struct Bench
{
    Args const& args;
    Op op;
    std::string name; // the field

    toy::Field& field_of(World& w, std::size_t const i) const
    {
        return name == "rho" ? w.ld.data[i].rho : w.ld.data[i].Bx;
    }

    void task(World& w, std::size_t const i, int const step) const
    {
        auto const& patch = w.ld.patches[i];
        work_on(field_of(w, i), patch, step, sweeps_of(args, patch));
    }

    // the simulated network's delay of a message from peer to this rank at step (seconds)
    void network(auto& c, int const step) const
    {
        if (args.latency <= 0 and args.jitter <= 0)
            return;
        c.simulate_latency([this, step](int const peer, std::size_t) {
            auto x = static_cast<std::uint64_t>(step) * 0x9E3779B97F4A7C15ull
                     ^ static_cast<std::uint64_t>(peer) * 0xBF58476D1CE4E5B9ull
                     ^ static_cast<std::uint64_t>(rank_of()) * 0x94D049BB133111EBull;
            x ^= x >> 31;
            x *= 0xD6E8FEB86659FD93ull;
            x ^= x >> 32;
            auto const u = static_cast<double>(x % 1000000u) / 1e6;
            return 1e-6 * (args.latency + args.jitter * u);
        });
    }

    // the work, then the copy (mpi::copy's steps): {work s, copy s}
    std::pair<double, double> whole_patch(World& w, int const step) const
    {
        auto const t0 = Clock::now();
        core::ThreadPool::INSTANCE().patch_exec(
            w.ld.patches.size(), [](std::size_t const i) { return i; },
            [&](std::size_t const i) { task(w, i, step); });
        auto const t1 = Clock::now();
        {
            mpi::Copy c{op, w.n, same_names({name}), by_name, w.ld};
            network(c, step);
            c.finish();
        }
        auto const t2 = Clock::now();
        return {std::chrono::duration<double>(t1 - t0).count(),
                std::chrono::duration<double>(t2 - t1).count()};
    }

    void distance_first_overlaps(World& w, int const step) const
    {
        mpi::Copy c{op, w.n, same_names({name}), by_name, w.ld};
        network(c, step);
        auto const order = c.remote_first();
        core::ThreadPool::INSTANCE().patch_exec(
            w.ld.patches.size(), [](std::size_t const i) { return i; },
            [&](std::size_t const i) { task(w, i, step); },
            core::ThreadPool::PatchHooks{.order   = order,
                                         .on_done = [&](std::size_t const i) { c.done(i); },
                                         .idle    = [&] { c.poll(); }});
        c.finish();
    }

    // false if any value differs between the worlds (NaN equal to NaN)
    bool same(World& a, World& b) const
    {
        bool ok = true;
        for (std::size_t i = 0; i < a.ld.patches.size(); ++i)
        {
            auto const& x = field_of(a, i).v;
            auto const& y = field_of(b, i).v;
            for (std::size_t k = 0; k < x.size(); ++k)
                ok = ok
                     and (std::bit_cast<std::uint64_t>(x[k]) == std::bit_cast<std::uint64_t>(y[k])
                          or (std::isnan(x[k]) and std::isnan(y[k])));
        }
        int local = ok, all = 0;
        MPI_Allreduce(&local, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return all;
    }

    int run() const
    {
        bool const run_whole   = args.mode != "distance_first_overlaps";
        bool const run_overlap = args.mode != "whole_patch";
        World a{args}, b{args};
        std::vector<double> whole, overlap, work, copy;
        int constexpr warm = 3;
        for (int step = 0; step < warm + args.steps; ++step)
        {
            if (run_whole)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                auto const t0       = Clock::now();
                auto const [ws, cs] = whole_patch(a, step);
                auto const wt
                    = max_over_ranks(std::chrono::duration<double>(Clock::now() - t0).count());
                auto const wk = max_over_ranks(ws), cp = max_over_ranks(cs);
                if (step >= warm)
                {
                    whole.push_back(wt);
                    work.push_back(wk);
                    copy.push_back(cp);
                }
            }
            if (run_overlap)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                auto const t1 = Clock::now();
                distance_first_overlaps(b, step);
                auto const ot
                    = max_over_ranks(std::chrono::duration<double>(Clock::now() - t1).count());
                if (step >= warm)
                    overlap.push_back(ot);
            }
        }
        bool const agree = !(run_whole and run_overlap) or same(a, b);

        auto const median = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        auto const least
            = [](std::vector<double> const& v) { return *std::min_element(v.begin(), v.end()); };
        if (rank_of() == 0)
        {
            std::size_t patches = a.h.levels[0].patches.size();
            std::printf(
                "copy_overlap: %d ranks x %zu pools x %zu threads, %d^2 cells, patches %d^2 "
                "(%zu), work %d imbalance %.2f, copy %s, latency %.0f + %.0f us, %d steps\n",
                size_of(), args.pools, args.threads, args.cells, args.patch, patches, args.work,
                args.imbalance, args.copy.c_str(), args.latency, args.jitter, args.steps);
            if (run_whole)
                std::printf("  %-24s median %8.3f ms  min %8.3f ms  (work %.3f ms, copy %.3f ms)\n",
                            "whole_patch", 1e3 * median(whole), 1e3 * least(whole),
                            1e3 * median(work), 1e3 * median(copy));
            if (run_overlap)
                std::printf("  %-24s median %8.3f ms  min %8.3f ms\n", "distance_first_overlaps",
                            1e3 * median(overlap), 1e3 * least(overlap));
            if (run_whole and run_overlap)
                std::printf("  speedup %.3fx, fields bit equal: %s\n",
                            median(whole) / median(overlap), agree ? "yes" : "NO");
        }
        return agree ? 0 : 1;
    }
};

} // namespace

int main(int argc, char** argv)
{
    int provided = 0; // MPI from the main thread only, the pools beside it
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rc = 0;
    try
    {
        auto const args = parse(argc, argv);
        damn::core::ThreadPool::INSTANCE().configure(args.pools, args.threads);
        rc = args.copy == "sum" ? Bench{args, damn::amr::sum_borders, "rho"}.run()
                                : Bench{args, damn::amr::fill_ghosts, "Bx"}.run();
        damn::core::ThreadPool::INSTANCE().stop();
    }
    catch (std::exception const& e)
    {
        std::fprintf(stderr, "copy_overlap: %s\n", e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return rc;
}
