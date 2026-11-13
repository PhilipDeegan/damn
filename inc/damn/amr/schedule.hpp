#ifndef DAMN_AMR_SCHEDULE_HPP
#define DAMN_AMR_SCHEDULE_HPP

// same level schedules: which values of which patches go where, for a set of resources
//
//   PatchGhost:  (BoxData) ghosts from neighbours' domains, periodic images included; a
//                patch's own domain points are never overwritten
//   BorderSum:   (BoxData) where ghost boxes overlap, every patch ends with the sum of all
//                patches' values (e.g. deposited moments)
//   BorderMax:   (BoxData) as BorderSum, with the max
//   GhostToDomain: (ParticleData) items a patch holds outside its domain (e.g. particles
//                that left it) are appended to the domain of the patch that owns their
//                cell, periodic images included; the source arrays are not cleared
//
// resources come in (source, destination) pairs; for BoxData both are the same resource
// a schedule is built for one level from the global patch list, and run on one rank's
// LevelData: everything is first read (packed), then written, so sums and maxes use the
// values from before the exchange
// single rank for now: transactions with a source on another rank throw when run

#include "damn/core/def.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/amr/box_data.hpp"
#include "damn/amr/particle_data.hpp"
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
#include <format>

namespace damn::amr
{
enum class Transfer : std::uint8_t { PatchGhost = 0, BorderSum, BorderMax, GhostToDomain, COUNT };

NO_DISCARD constexpr bool is_particle_transfer(Transfer const kind)
{
    return kind == Transfer::GhostToDomain;
}

// what moves from where to where; for BoxData transfers src == dst
struct ResourcePair
{
    std::string src{};
    std::string dst{};

    bool operator==(ResourcePair const&) const = default;
};

NO_DISCARD inline std::vector<ResourcePair> same_names(std::vector<std::string> const& names)
{
    std::vector<ResourcePair> pairs;
    for (auto const& name : names)
        pairs.push_back({name, name});
    return pairs;
}

template<std::size_t dim>
struct Transaction
{
    std::size_t dst = 0;           // local patch index in the LevelData
    Patch<dim> src{};              // source patch, any rank
    ResourcePair names{};          // source and destination resources
    core::Box<int, dim> box{};     // dst points (or cells), level index space
    core::Point<int, dim> shift{}; // the source of dst point p is at p - shift
};

template<std::size_t dim>
struct Schedule
{
    Transfer kind = Transfer::PatchGhost;
    int level     = 0;
    int rank      = 0; // whose LevelData it runs on
    std::vector<Transaction<dim>> transactions{};
    std::vector<Patch<dim>> built_on{}; // the level's patches when built
};


namespace detail
{
    // fn(resource) for the resource called name, BoxData or ParticleData
    template<std::size_t dim, bool particles>
    void with_resource_of_kind(auto& patch_data, std::string const& name, auto&& fn)
    {
        bool const found = with_resource(patch_data, name, [&](auto& resource) {
            using R = std::decay_t<decltype(resource)>;
            if constexpr (particles)
            {
                if constexpr (ParticleData<R, dim>)
                    fn(resource);
                else
                    throw std::runtime_error("damn::amr schedule: " + name
                                             + " is not ParticleData");
            }
            else
            {
                if constexpr (BoxData<R, dim>)
                    fn(resource);
                else
                    throw std::runtime_error("damn::amr schedule: " + name + " is not BoxData");
            }
        });
        if (!found)
            throw std::runtime_error("damn::amr schedule: no resource " + name);
    }

    template<std::size_t dim>
    void with_box_data(auto& patch_data, std::string const& name, auto&& fn)
    {
        with_resource_of_kind<dim, false>(patch_data, name, fn);
    }

    template<std::size_t dim>
    void with_particle_data(auto& patch_data, std::string const& name, auto&& fn)
    {
        with_resource_of_kind<dim, true>(patch_data, name, fn);
    }

    template<std::size_t dim>
    void add_transactions(std::vector<Transaction<dim>>& out, std::size_t const dst,
                          Patch<dim> const& src, ResourcePair const& names,
                          core::Point<int, dim> const& shift,
                          std::vector<core::Box<int, dim>> const& boxes)
    {
        for (auto const& box : boxes)
            out.push_back({dst, src, names, box, shift});
    }

    template<typename T>
    void push_bytes(std::vector<std::byte>& buf, T const& v)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        auto const at = buf.size();
        buf.resize(at + sizeof(T));
        std::memcpy(buf.data() + at, &v, sizeof(T));
    }

    template<typename T>
    T read_bytes(std::vector<std::byte> const& buf, std::size_t& at)
    {
        T v;
        std::memcpy(&v, buf.data() + at, sizeof(T));
        at += sizeof(T);
        return v;
    }
} // namespace detail


// the schedule of `kind` for resource pairs of level ln, run on rank's LevelData ld
template<std::size_t dim, typename PatchData>
NO_DISCARD Schedule<dim> make_schedule(Transfer const kind, Hierarchy<dim> const& h,
                                       int const ln, LevelData<dim, PatchData>& ld,
                                       int const rank, std::vector<ResourcePair> const& pairs)
{
    if (!level_exists(h, ln) or ld.level != ln)
        throw std::runtime_error("damn::amr::make_schedule: level mismatch");

    auto const& level = h.levels[ln];

    std::vector<core::Point<int, dim>> shifts{core::Point<int, dim>{}};
    for (auto const& s : periodic_shifts(h, ln))
        shifts.push_back(s);

    Schedule<dim> schedule{kind, ln, rank, {}, level.patches};

    // fn(src patch, shift) for every source image but the destination itself
    auto const for_each_source = [&](auto const& dst, auto&& fn) {
        for (auto const& src : level.patches)
            for (auto const& shift : shifts)
                if (src.id != dst.id or shift != core::Point<int, dim>{})
                    fn(src, shift);
    };

    for (std::size_t d = 0; d < size(ld); ++d)
    {
        auto const& dst = ld.patches[d];
        for (auto const& pair : pairs)
        {
            if (is_particle_transfer(kind))
            {
                // the destination's resource gives the boxes of any patch
                detail::with_particle_data<dim>(ld.data[d], pair.dst, [&](auto const& res) {
                    auto const dst_domain = domain_box(res, dst.box);
                    for_each_source(dst, [&](auto const& src, auto const& shift) {
                        if (auto const overlap
                            = dst_domain * core::shift(ghost_box(res, src.box), shift))
                            detail::add_transactions(schedule.transactions, d, src, pair,
                                                     shift, {*overlap});
                    });
                });
                continue;
            }

            detail::with_box_data<dim>(ld.data[d], pair.dst, [&](auto const& res) {
                auto const dst_ghost  = ghost_box(res, dst.box);
                auto const dst_domain = domain_box(res, dst.box);

                for_each_source(dst, [&](auto const& src, auto const& shift) {
                    auto const src_box = kind == Transfer::PatchGhost ? domain_box(res, src.box)
                                                                      : ghost_box(res, src.box);
                    auto const overlap = dst_ghost * core::shift(src_box, shift);
                    if (!overlap)
                        return;

                    if (kind == Transfer::PatchGhost) // never into dst's own domain
                        detail::add_transactions(schedule.transactions, d, src, pair, shift,
                                                 overlap->remove(dst_domain));
                    else
                        detail::add_transactions(schedule.transactions, d, src, pair, shift,
                                                 {*overlap});
                });
            });
        }
    }

    return schedule;
}

// same name on both sides
template<std::size_t dim, typename PatchData>
NO_DISCARD Schedule<dim> make_schedule(Transfer const kind, Hierarchy<dim> const& h,
                                       int const ln, LevelData<dim, PatchData>& ld,
                                       int const rank, std::vector<std::string> const& names)
{
    return make_schedule(kind, h, ln, ld, rank, same_names(names));
}


// true if level still has the patches the schedule was built on
template<std::size_t dim>
NO_DISCARD bool is_current(Schedule<dim> const& schedule, Level<dim> const& level)
{
    return schedule.level == level.number and schedule.built_on == level.patches;
}


namespace detail
{
    template<std::size_t dim, typename PatchData>
    std::vector<std::byte> pack(Schedule<dim> const& schedule, Transaction<dim> const& tr,
                                LevelData<dim, PatchData>& ld)
    {
        if (tr.src.rank != schedule.rank)
            throw std::runtime_error("damn::amr::run: remote patches need MPI, not done yet");

        auto& src_data = ld.data[local_index(ld, tr.src.id)];
        std::vector<std::byte> buf;

        if (is_particle_transfer(schedule.kind))
            with_particle_data<dim>(src_data, tr.names.src, [&](auto const& src) {
                for_each_item(src, [&](auto const& item) {
                    if (isIn(cell(item) + tr.shift, tr.box))
                        push_bytes(buf, item);
                });
            });
        else
            with_box_data<dim>(src_data, tr.names.src, [&](auto const& src) {
                using T = box_data_value_t<decltype(src), dim>;
                buf.reserve(tr.box.size() * sizeof(T));
                core::for_each_point(tr.box, [&](auto const& p) {
                    push_bytes(buf, T{value(src, tr.src.box, p - tr.shift)});
                });
            });
        return buf;
    }

    template<std::size_t dim, typename PatchData>
    void unpack(Schedule<dim> const& schedule, Transaction<dim> const& tr,
                LevelData<dim, PatchData>& ld, std::vector<std::byte> const& buf)
    {
        auto& dst_data      = ld.data[tr.dst];
        auto const& dst_box = ld.patches[tr.dst].box;
        std::size_t at      = 0;

        if (is_particle_transfer(schedule.kind))
            with_particle_data<dim>(dst_data, tr.names.dst, [&](auto& dst) {
                using Item = typename std::decay_t<decltype(dst)>::value_type;
                while (at < buf.size())
                    append(dst, read_bytes<Item>(buf, at), tr.shift);
            });
        else
            with_box_data<dim>(dst_data, tr.names.dst, [&](auto& dst) {
                using T = box_data_value_t<decltype(dst), dim>;
                core::for_each_point(tr.box, [&](auto const& p) {
                    auto const v = read_bytes<T>(buf, at);
                    auto& d      = value(dst, dst_box, p);
                    if (schedule.kind == Transfer::PatchGhost)
                        d = v;
                    else if (schedule.kind == Transfer::BorderSum)
                        d += v;
                    else
                        d = std::max(d, v);
                });
            });
    }
} // namespace detail


// runs schedule on ld: reads every source region, then writes every destination region
template<std::size_t dim, typename PatchData>
void run(Schedule<dim> const& schedule, LevelData<dim, PatchData>& ld)
{
    if (ld.level != schedule.level)
        throw std::runtime_error("damn::amr::run: level mismatch");

    std::vector<std::vector<std::byte>> buffers;
    buffers.reserve(schedule.transactions.size());
    for (auto const& tr : schedule.transactions)
        buffers.emplace_back(detail::pack(schedule, tr, ld));

    for (std::size_t t = 0; t < schedule.transactions.size(); ++t)
        detail::unpack(schedule, schedule.transactions[t], ld, buffers[t]);
}

} // namespace damn::amr

#endif // DAMN_AMR_SCHEDULE_HPP
