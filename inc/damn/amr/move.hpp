#ifndef DAMN_AMR_MOVE_HPP
#define DAMN_AMR_MOVE_HPP

// moving data between the patches of a level, along Neighbours' links
//
//   a move is what goes where (Region) and what happens on arrival (Arrival):
//
//     fill_ghosts     Ghosts,  Assign   (BoxData) neighbours' domain values into my ghosts;
//                                       my own domain points are never written
//     sum_borders     Borders, Sum      (BoxData) where ghost boxes overlap, every patch ends
//                                       with the sum of all patches' values (deposits)
//     max_borders     Borders, Max      as sum_borders, with the max
//     gather_leaving  Leaving, Append   (ParticleData) items a neighbour holds outside its
//                                       domain are appended to the domain of the patch owning
//                                       their cell; the source arrays are not cleared
//
//   move(fill_ghosts, neighbours, {"Bx", "By"}, level_data)
//
// resources come as (source, destination) pairs (ResourcePair), or names for fields. What
// moves where is planned on first use for a region and set of resources, and kept in the
// Neighbours. How each piece moves follows from its link's Distance:
//   Self, Local:  copied directly, patch to patch; sums and maxes go through one buffer kept
//                 in the plan, so they use the values from before the move
//   Numa:         as Local for now
//   Node, Remote: need MPI, not done yet: throws

#include "damn/core/def.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/amr/box_data.hpp"
#include "damn/amr/particle_data.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/topology.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace damn::amr
{
enum class Arrival : std::uint8_t { Assign, Sum, Max, Append };

template<Region region_, Arrival arrival_>
struct Move
{
    auto static constexpr region  = region_;
    auto static constexpr arrival = arrival_;

    static_assert((region == Region::Ghosts and arrival == Arrival::Assign)
                      or (region == Region::Borders
                          and (arrival == Arrival::Sum or arrival == Arrival::Max))
                      or (region == Region::Leaving and arrival == Arrival::Append),
                  "damn::amr::Move: not a supported region and arrival");
};

inline constexpr Move<Region::Ghosts, Arrival::Assign> fill_ghosts{};
inline constexpr Move<Region::Borders, Arrival::Sum> sum_borders{};
inline constexpr Move<Region::Borders, Arrival::Max> max_borders{};
inline constexpr Move<Region::Leaving, Arrival::Append> gather_leaving{};


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
                    throw std::runtime_error("damn::amr::move: " + name + " is not ParticleData");
            }
            else
            {
                if constexpr (BoxData<R, dim>)
                    fn(resource);
                else
                    throw std::runtime_error("damn::amr::move: " + name + " is not BoxData");
            }
        });
        if (!found)
            throw std::runtime_error("damn::amr::move: no resource " + name);
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
    NO_DISCARD bool contains(core::Box<int, dim> const& outer, core::Box<int, dim> const& inner)
    {
        auto const overlap = outer * inner;
        return overlap and *overlap == inner;
    }

    // the links of a resource's boxes reach past: the neighbours were built too short
    template<std::size_t dim>
    void check_reach(Neighbours<dim> const& n, std::string const& name,
                     core::Box<int, dim> const& patch_box,
                     core::Box<int, dim> const& resource_ghost_box)
    {
        if (!contains(core::grow(patch_box, n.reach), resource_ghost_box))
            throw std::runtime_error("damn::amr::move: " + name
                                     + " reaches past the neighbours' reach of "
                                     + std::to_string(n.reach));
    }

    template<std::size_t dim, typename PatchData>
    NO_DISCARD Plan<dim> make_plan(Region const region, Neighbours<dim> const& n,
                                   LevelData<dim, PatchData>& ld,
                                   std::vector<ResourcePair> const& pairs)
    {
        Plan<dim> plan;
        std::size_t bytes = 0;

        for (std::size_t d = 0; d < n.local.size(); ++d)
        {
            auto const& dst = n.local[d];
            for (auto const& pair : pairs)
            {
                if (region == Region::Leaving)
                {
                    // the destination's resource gives the boxes of any patch
                    with_particle_data<dim>(ld.data[d], pair.dst, [&](auto const& res) {
                        check_reach(n, pair.dst, dst.box, ghost_box(res, dst.box));
                        auto const dst_domain = domain_box(res, dst.box);
                        for (auto const& link : n.of[d])
                            if (auto const overlap
                                = dst_domain
                                  * core::shift(ghost_box(res, link.patch.box), link.shift))
                                plan.copies.push_back({d, link, pair, *overlap});
                    });
                    continue;
                }

                with_box_data<dim>(ld.data[d], pair.dst, [&](auto const& res) {
                    using T               = box_data_value_t<decltype(res), dim>;
                    auto const dst_ghost  = ghost_box(res, dst.box);
                    auto const dst_domain = domain_box(res, dst.box);
                    check_reach(n, pair.dst, dst.box, dst_ghost);

                    for (auto const& link : n.of[d])
                    {
                        auto const src_box = region == Region::Ghosts
                                                 ? domain_box(res, link.patch.box)
                                                 : ghost_box(res, link.patch.box);
                        auto const overlap = dst_ghost * core::shift(src_box, link.shift);
                        if (!overlap)
                            continue;

                        if (region == Region::Ghosts) // never into dst's own domain
                        {
                            for (auto const& box : overlap->remove(dst_domain))
                                plan.copies.push_back({d, link, pair, box});
                            continue;
                        }
                        // Borders: through the buffer
                        auto const size = static_cast<std::size_t>(overlap->size()) * sizeof(T);
                        plan.copies.push_back({d, link, pair, *overlap, bytes, size});
                        bytes += size;
                    }
                });
            }
        }
        plan.buffer.resize(bytes);
        return plan;
    }

    template<std::size_t dim>
    void check_distance(Copy<dim> const& c)
    {
        if (c.src.local == not_local or c.src.distance >= Distance::Node)
            throw std::runtime_error("damn::amr::move: patch " + std::to_string(c.src.patch.id)
                                     + " is " + std::string{to_string(c.src.distance)}
                                     + ", moves between ranks need MPI, not done yet");
    }

    template<typename T>
    T read(std::byte const* const at)
    {
        T v;
        std::memcpy(&v, at, sizeof(T));
        return v;
    }

    template<std::size_t dim, typename PatchData>
    void assign(Plan<dim> const& plan, LevelData<dim, PatchData>& ld)
    {
        for (auto const& c : plan.copies)
            with_box_data<dim>(ld.data[c.src.local], c.names.src, [&](auto const& src) {
                with_box_data<dim>(ld.data[c.dst], c.names.dst, [&](auto& dst) {
                    auto const& dst_box = ld.patches[c.dst].box;
                    core::for_each_point(c.box, [&](auto const& p) {
                        value(dst, dst_box, p) = value(src, c.src.patch.box, p - c.src.shift);
                    });
                });
            });
    }

    // every source read into the buffer, then every destination written
    template<Arrival arrival, std::size_t dim, typename PatchData>
    void through_buffer(Plan<dim>& plan, LevelData<dim, PatchData>& ld)
    {
        for (auto const& c : plan.copies)
            with_box_data<dim>(ld.data[c.src.local], c.names.src, [&](auto const& src) {
                using T  = box_data_value_t<decltype(src), dim>;
                auto* at = plan.buffer.data() + c.offset;
                core::for_each_point(c.box, [&](auto const& p) {
                    T const v{value(src, c.src.patch.box, p - c.src.shift)};
                    std::memcpy(at, &v, sizeof(T));
                    at += sizeof(T);
                });
            });

        for (auto const& c : plan.copies)
            with_box_data<dim>(ld.data[c.dst], c.names.dst, [&](auto& dst) {
                using T             = box_data_value_t<decltype(dst), dim>;
                auto const& dst_box = ld.patches[c.dst].box;
                auto const* at      = plan.buffer.data() + c.offset;
                core::for_each_point(c.box, [&](auto const& p) {
                    auto const v = read<T>(at);
                    at += sizeof(T);
                    auto& d = value(dst, dst_box, p);
                    if constexpr (arrival == Arrival::Sum)
                        d += v;
                    else
                        d = std::max(d, v);
                });
            });
    }

    template<std::size_t dim, typename PatchData>
    void append(Plan<dim> const& plan, LevelData<dim, PatchData>& ld)
    {
        for (auto const& c : plan.copies)
            with_particle_data<dim>(ld.data[c.src.local], c.names.src, [&](auto const& src) {
                with_particle_data<dim>(ld.data[c.dst], c.names.dst, [&](auto& dst) {
                    using Item = typename std::decay_t<decltype(dst)>::value_type;
                    auto const leaving
                        = [&](auto const& item) { return isIn(cell(item) + c.src.shift, c.box); };

                    if (static_cast<void const*>(&src) != static_cast<void const*>(&dst))
                    {
                        for_each_item(src, [&](auto const& item) {
                            if (leaving(item))
                                append(dst, item, c.src.shift);
                        });
                        return;
                    }
                    // the same array: collect first, appending invalidates it
                    std::vector<Item> items;
                    for_each_item(src, [&](auto const& item) {
                        if (leaving(item))
                            items.push_back(item);
                    });
                    for (auto const& item : items)
                        append(dst, item, c.src.shift);
                });
            });
    }
} // namespace detail


// the plan of a region for resource pairs, made on first use
template<std::size_t dim, typename PatchData>
Plan<dim>& plan_for(Region const region, Neighbours<dim>& n, LevelData<dim, PatchData>& ld,
                    std::vector<ResourcePair> const& pairs)
{
    if (ld.level != n.level or ld.patches != n.local)
        throw std::runtime_error("damn::amr::move: the neighbours are not the level data's"
                                 " (rebuild them after a regrid)");

    typename Neighbours<dim>::PlanKey key{region, pairs};
    auto found = n.plans.find(key);
    if (found == n.plans.end())
        found = n.plans.emplace(std::move(key), detail::make_plan(region, n, ld, pairs)).first;
    return found->second;
}

template<Region region, Arrival arrival, std::size_t dim, typename PatchData>
void move(Move<region, arrival>, Neighbours<dim>& n, std::vector<ResourcePair> const& pairs,
          LevelData<dim, PatchData>& ld)
{
    auto& plan = plan_for(region, n, ld, pairs);
    for (auto const& c : plan.copies)
        detail::check_distance(c);

    if constexpr (arrival == Arrival::Assign)
        detail::assign(plan, ld);
    else if constexpr (arrival == Arrival::Append)
        detail::append(plan, ld);
    else
        detail::through_buffer<arrival>(plan, ld);
}

// fields: the same name on both sides
template<Region region, Arrival arrival, std::size_t dim, typename PatchData>
void move(Move<region, arrival> const m, Neighbours<dim>& n, std::vector<std::string> const& names,
          LevelData<dim, PatchData>& ld)
{
    move(m, n, same_names(names), ld);
}

} // namespace damn::amr

#endif // DAMN_AMR_MOVE_HPP
