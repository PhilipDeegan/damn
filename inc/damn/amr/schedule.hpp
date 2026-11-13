#ifndef DAMN_AMR_SCHEDULE_HPP
#define DAMN_AMR_SCHEDULE_HPP

// same level schedules: which values of which patches go where, for a set of resources
//
//   PatchGhost: ghosts from neighbours' domains (periodic images included); a patch's
//               own domain points are never overwritten
//   BorderSum:  where ghost boxes overlap, every patch ends with the sum of all patches'
//               values (e.g. deposited moments)
//   BorderMax:  as BorderSum, with the max
//
// a schedule is built for one level from the global patch list, and run on one rank's
// LevelData: values are first all read (packed), then all written, so sums and maxes use
// the values from before the exchange
// single rank for now: transactions with a source on another rank throw when run

#include "damn/core/def.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/amr/box_data.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/level_data.hpp"

#include <vector>
#include <string>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace damn::amr
{
enum class Transfer : std::uint8_t { PatchGhost = 0, BorderSum, BorderMax, COUNT };

template<std::size_t dim>
struct Transaction
{
    std::size_t dst = 0;       // local patch index in the LevelData
    Patch<dim> src{};          // source patch, any rank
    std::string name{};        // resource, the same on both sides
    core::Box<int, dim> box{}; // dst points, level index space
    core::Point<int, dim> shift{}; // the source value of dst point p is at p - shift
};

template<std::size_t dim>
struct Schedule
{
    Transfer kind  = Transfer::PatchGhost;
    int level      = 0;
    int rank       = 0; // whose LevelData it runs on
    std::vector<Transaction<dim>> transactions{};
    std::vector<Patch<dim>> built_on{}; // the level's patches when built
};


namespace detail
{
    template<std::size_t dim>
    void throw_not_box_data(std::string const& name)
    {
        throw std::runtime_error("damn::amr schedule: resource " + name + " is not BoxData");
    }

    // fn(resource) for the BoxData resource called name
    template<std::size_t dim>
    void with_box_data(auto& patch_data, std::string const& name, auto&& fn)
    {
        bool const found = with_resource(patch_data, name, [&](auto& resource) {
            using R = std::decay_t<decltype(resource)>;
            if constexpr (BoxData<R, dim>)
                fn(resource);
            else
                throw_not_box_data<dim>(name);
        });
        if (!found)
            throw std::runtime_error("damn::amr schedule: no resource " + name);
    }

    template<std::size_t dim>
    void add_transactions(std::vector<Transaction<dim>>& out, std::size_t const dst,
                          Patch<dim> const& src, std::string const& name,
                          core::Point<int, dim> const& shift,
                          std::vector<core::Box<int, dim>> const& boxes)
    {
        for (auto const& box : boxes)
            out.push_back({dst, src, name, box, shift});
    }
} // namespace detail


// the schedule of `kind` for resources `names` of level ln, run on rank's LevelData ld
template<std::size_t dim, typename PatchData>
NO_DISCARD Schedule<dim> make_schedule(Transfer const kind, Hierarchy<dim> const& h,
                                       int const ln, LevelData<dim, PatchData>& ld,
                                       int const rank, std::vector<std::string> const& names)
{
    if (!level_exists(h, ln) or ld.level != ln)
        throw std::runtime_error("damn::amr::make_schedule: level mismatch");

    auto const& level = h.levels[ln];

    std::vector<core::Point<int, dim>> shifts{core::Point<int, dim>{}};
    for (auto const& s : periodic_shifts(h, ln))
        shifts.push_back(s);

    Schedule<dim> schedule{kind, ln, rank, {}, level.patches};

    for (std::size_t d = 0; d < size(ld); ++d)
    {
        auto const& dst = ld.patches[d];
        for (auto const& name : names)
            detail::with_box_data<dim>(ld.data[d], name, [&](auto const& resource) {
                auto const dst_ghost  = ghost_box(resource, dst.box);
                auto const dst_domain = domain_box(resource, dst.box);

                for (auto const& src : level.patches)
                    for (auto const& shift : shifts)
                    {
                        if (src.id == dst.id and shift == core::Point<int, dim>{})
                            continue; // itself

                        auto const src_box = kind == Transfer::PatchGhost
                                                 ? domain_box(resource, src.box)
                                                 : ghost_box(resource, src.box);
                        auto const overlap = dst_ghost * core::shift(src_box, shift);
                        if (!overlap)
                            continue;

                        if (kind == Transfer::PatchGhost) // never into dst's own domain
                            detail::add_transactions(schedule.transactions, d, src, name, shift,
                                                     overlap->remove(dst_domain));
                        else
                            detail::add_transactions(schedule.transactions, d, src, name, shift,
                                                     {*overlap});
                    }
            });
    }

    return schedule;
}


// true if level still has the patches the schedule was built on
template<std::size_t dim>
NO_DISCARD bool is_current(Schedule<dim> const& schedule, Level<dim> const& level)
{
    return schedule.level == level.number and schedule.built_on == level.patches;
}


// runs schedule on ld: reads every source region, then writes every destination region
template<std::size_t dim, typename PatchData>
void run(Schedule<dim> const& schedule, LevelData<dim, PatchData>& ld)
{
    if (ld.level != schedule.level)
        throw std::runtime_error("damn::amr::run: level mismatch");

    // pack: the source values of each transaction, as bytes
    std::vector<std::vector<std::byte>> buffers(schedule.transactions.size());
    for (std::size_t t = 0; t < schedule.transactions.size(); ++t)
    {
        auto const& tr = schedule.transactions[t];
        if (tr.src.rank != schedule.rank)
            throw std::runtime_error("damn::amr::run: remote patches need MPI, not done yet");

        detail::with_box_data<dim>(ld.data[local_index(ld, tr.src.id)], tr.name,
                                   [&](auto const& src) {
                                       using T = box_data_value_t<decltype(src), dim>;
                                       static_assert(std::is_trivially_copyable_v<T>);
                                       auto& buf = buffers[t];
                                       buf.resize(tr.box.size() * sizeof(T));
                                       std::size_t i = 0;
                                       core::for_each_point(tr.box, [&](auto const& p) {
                                           T const v = value(src, tr.src.box, p - tr.shift);
                                           std::memcpy(buf.data() + i, &v, sizeof(T));
                                           i += sizeof(T);
                                       });
                                   });
    }

    // unpack: write or combine into the destinations
    for (std::size_t t = 0; t < schedule.transactions.size(); ++t)
    {
        auto const& tr      = schedule.transactions[t];
        auto const& dst_box = ld.patches[tr.dst].box;
        detail::with_box_data<dim>(ld.data[tr.dst], tr.name, [&](auto& dst) {
            using T          = box_data_value_t<decltype(dst), dim>;
            auto const& buf  = buffers[t];
            std::size_t i    = 0;
            core::for_each_point(tr.box, [&](auto const& p) {
                T v;
                std::memcpy(&v, buf.data() + i, sizeof(T));
                i += sizeof(T);
                auto& d = value(dst, dst_box, p);
                if (schedule.kind == Transfer::PatchGhost)
                    d = v;
                else if (schedule.kind == Transfer::BorderSum)
                    d += v;
                else
                    d = std::max(d, v);
            });
        });
    }
}

} // namespace damn::amr

#endif // DAMN_AMR_SCHEDULE_HPP
