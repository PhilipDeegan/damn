#ifndef DAMN_AMR_COPY_HPP
#define DAMN_AMR_COPY_HPP

// copying data between the patches of a level, along Neighbours' links: sources are never
// modified (cleared or overwritten)
//
//   a copy is what goes where (Region) and what happens on arrival (Arrival):
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
//   copy(fill_ghosts, neighbours, {"Bx", "By"}, level_data)
//
// resources come as (source, destination) pairs (ResourcePair), or names for fields. What
// goes where is planned on first use for a region and set of resources, and kept in the
// Neighbours. How each piece goes follows from its link's Distance:
//   Self, Local:  directly, patch to patch; sums and maxes go through one buffer kept in the
//                 plan, so they use the values from before the copy
//   Numa:         as Local for now
//   Node, Remote: copy() here throws: damn/mpi/copy.hpp's damn::mpi::copy sends them, with
//                 the pieces packed by pack_piece and unpacked by unpack_piece below

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
template<std::size_t dim>
struct Patch;

enum class Arrival : std::uint8_t { Assign, Sum, Max, Append };

template<Region region_, Arrival arrival_>
struct CopyOp
{
    auto static constexpr region  = region_;
    auto static constexpr arrival = arrival_;

    static_assert((region == Region::Ghosts and arrival == Arrival::Assign)
                      or (region == Region::Borders
                          and (arrival == Arrival::Sum or arrival == Arrival::Max))
                      or (region == Region::Leaving and arrival == Arrival::Append),
                  "damn::amr::CopyOp: not a supported region and arrival");
};

inline constexpr CopyOp<Region::Ghosts, Arrival::Assign> fill_ghosts{};
inline constexpr CopyOp<Region::Borders, Arrival::Sum> sum_borders{};
inline constexpr CopyOp<Region::Borders, Arrival::Max> max_borders{};
inline constexpr CopyOp<Region::Leaving, Arrival::Append> gather_leaving{};


namespace detail
{
    // fn(resource) for the resource called name, BoxData or ParticleData
    template<std::size_t dim, bool particles, typename PatchData, typename Fn>
    void with_resource_of_kind(PatchData& patch_data, std::string const& name, Fn&& fn);

    template<std::size_t dim, typename PatchData, typename Fn>
    void with_box_data(PatchData& patch_data, std::string const& name, Fn&& fn);

    template<std::size_t dim, typename PatchData, typename Fn>
    void with_particle_data(PatchData& patch_data, std::string const& name, Fn&& fn);

    template<std::size_t dim>
    NO_DISCARD bool contains(core::Box<int, dim> const& outer, core::Box<int, dim> const& inner);

    // throws if a resource's ghost box reaches past the neighbours' reach
    template<std::size_t dim>
    void check_reach(Neighbours<dim> const& n, std::string const& name,
                     core::Box<int, dim> const& patch_box,
                     core::Box<int, dim> const& resource_ghost_box);

    // the pieces of leaving items (Region::Leaving): ParticleData
    template<std::size_t dim, typename PatchData>
    NO_DISCARD Plan<dim> particle_plan(Neighbours<dim> const& n, LevelData<dim, PatchData>& ld,
                                       std::vector<ResourcePair> const& pairs);

    // the pieces of ghosts or borders: BoxData; local borders through the plan's buffer
    template<std::size_t dim, typename PatchData>
    NO_DISCARD Plan<dim> box_plan(Region region, Neighbours<dim> const& n,
                                  LevelData<dim, PatchData>& ld,
                                  std::vector<ResourcePair> const& pairs);

    template<std::size_t dim>
    NO_DISCARD bool is_local(Piece<dim> const& piece);

    // throws if the plan has pieces to or from another rank
    template<std::size_t dim>
    void check_local(Plan<dim> const& plan);

    template<typename T>
    NO_DISCARD T read(std::byte const*& at);

    template<typename T>
    void write(std::vector<std::byte>& out, T const& v);

    // local pieces only, below: pieces from or to other ranks are damn::mpi::copy's
    template<std::size_t dim, typename PatchData>
    void assign(Plan<dim> const& plan, LevelData<dim, PatchData>& ld);

    template<std::size_t dim, typename PatchData>
    void append_items(Plan<dim> const& plan, LevelData<dim, PatchData>& ld);

    // sums and maxes: every local source read into the buffer, then (unpack) written
    template<std::size_t dim, typename PatchData>
    void pack_local(Plan<dim>& plan, LevelData<dim, PatchData>& ld);

    template<Arrival arrival, std::size_t dim, typename PatchData>
    void unpack_local(Plan<dim>& plan, LevelData<dim, PatchData>& ld);
} // namespace detail


// a piece's data from its (local) source, appended to out: field values in the box's point
// order, or the number of items then the items
template<std::size_t dim, typename PatchData>
void pack_piece(Region region, Piece<dim> const& piece, LevelData<dim, PatchData>& ld,
                std::vector<std::byte>& out);

// a piece's data, packed by pack_piece on another rank, into its (local) destination; at
// moves past it
template<Arrival arrival, std::size_t dim, typename PatchData>
void unpack_piece(Region region, Piece<dim> const& piece, LevelData<dim, PatchData>& ld,
                  std::byte const*& at);


// the plan of a region for resource pairs, made on first use
template<std::size_t dim, typename PatchData>
Plan<dim>& plan_for(Region const region, Neighbours<dim>& n, LevelData<dim, PatchData>& ld,
                    std::vector<ResourcePair> const& pairs)
{
    if (ld.level != n.level or ld.patches != n.local)
        throw std::runtime_error("damn::amr::copy: the neighbours are not the level data's"
                                 " (rebuild them after a regrid)");

    typename Neighbours<dim>::PlanKey key{region, pairs};
    auto found = n.plans.find(key);
    if (found == n.plans.end())
        found = n.plans
                    .emplace(std::move(key), region == Region::Leaving
                                                 ? detail::particle_plan(n, ld, pairs)
                                                 : detail::box_plan(region, n, ld, pairs))
                    .first;
    return found->second;
}

// on one rank: throws if any piece comes from, or must go to, another rank
template<Region region, Arrival arrival, std::size_t dim, typename PatchData>
void copy(CopyOp<region, arrival>, Neighbours<dim>& n, std::vector<ResourcePair> const& pairs,
          LevelData<dim, PatchData>& ld)
{
    auto& plan = plan_for(region, n, ld, pairs);
    detail::check_local(plan);

    if constexpr (arrival == Arrival::Assign)
        detail::assign(plan, ld);
    else if constexpr (arrival == Arrival::Append)
        detail::append_items(plan, ld);
    else
    {
        detail::pack_local(plan, ld);
        detail::unpack_local<arrival>(plan, ld);
    }
}

// fields: the same name on both sides
template<Region region, Arrival arrival, std::size_t dim, typename PatchData>
void copy(CopyOp<region, arrival> const op, Neighbours<dim>& n,
          std::vector<std::string> const& names, LevelData<dim, PatchData>& ld)
{
    copy(op, n, same_names(names), ld);
}


// ---------------------------------------------------------------- detail

template<std::size_t dim, bool particles, typename PatchData, typename Fn>
void detail::with_resource_of_kind(PatchData& patch_data, std::string const& name, Fn&& fn)
{
    bool const found = with_resource(patch_data, name, [&](auto& resource) {
        using R = std::decay_t<decltype(resource)>;
        if constexpr (particles)
        {
            if constexpr (ParticleData<R, dim>)
                fn(resource);
            else
                throw std::runtime_error("damn::amr::copy: " + name + " is not ParticleData");
        }
        else
        {
            if constexpr (BoxData<R, dim>)
                fn(resource);
            else
                throw std::runtime_error("damn::amr::copy: " + name + " is not BoxData");
        }
    });
    if (!found)
        throw std::runtime_error("damn::amr::copy: no resource " + name);
}

template<std::size_t dim, typename PatchData, typename Fn>
void detail::with_box_data(PatchData& patch_data, std::string const& name, Fn&& fn)
{
    with_resource_of_kind<dim, false>(patch_data, name, fn);
}

template<std::size_t dim, typename PatchData, typename Fn>
void detail::with_particle_data(PatchData& patch_data, std::string const& name, Fn&& fn)
{
    with_resource_of_kind<dim, true>(patch_data, name, fn);
}

template<std::size_t dim>
bool detail::contains(core::Box<int, dim> const& outer, core::Box<int, dim> const& inner)
{
    auto const overlap = outer * inner;
    return overlap and *overlap == inner;
}

template<std::size_t dim>
void detail::check_reach(Neighbours<dim> const& n, std::string const& name,
                         core::Box<int, dim> const& patch_box,
                         core::Box<int, dim> const& resource_ghost_box)
{
    if (!contains(core::grow(patch_box, n.reach), resource_ghost_box))
        throw std::runtime_error("damn::amr::copy: " + name
                                 + " reaches past the neighbours' reach of "
                                 + std::to_string(n.reach));
}

// for a destination patch (local, or another rank's for outgoing pieces) and its links, with
// resources of the destination's kind (res, any patch's: boxes only depend on the patch box)
template<std::size_t dim, typename PatchData>
Plan<dim> detail::particle_plan(Neighbours<dim> const& n, LevelData<dim, PatchData>& ld,
                                std::vector<ResourcePair> const& pairs)
{
    Plan<dim> plan{Region::Leaving};
    auto const pieces = [&](auto const& res, Patch<dim> const& dst, std::size_t const d,
                            auto const& links, ResourcePair const& pair, auto& out) {
        auto const dst_domain = domain_box(res, dst.box);
        for (auto const& link : links)
            if (auto const overlap
                = dst_domain * core::shift(ghost_box(res, link.patch.box), link.shift))
                out.push_back({d, link, pair, *overlap, 0, 0, dst.rank});
    };

    for (auto const& pair : pairs)
    {
        for (std::size_t d = 0; d < n.local.size(); ++d)
            with_particle_data<dim>(ld.data[d], pair.dst, [&](auto const& res) {
                check_reach(n, pair.dst, n.local[d].box, ghost_box(res, n.local[d].box));
                pieces(res, n.local[d], d, n.of[d], pair, plan.pieces);
            });
        if (!n.remote.empty())
            with_particle_data<dim>(ld.data[0], pair.dst, [&](auto const& res) {
                for (auto const& [dst, links] : n.remote)
                    pieces(res, dst, not_local, links, pair, plan.outgoing);
            });
    }
    return plan;
}

template<std::size_t dim, typename PatchData>
Plan<dim> detail::box_plan(Region const region, Neighbours<dim> const& n,
                           LevelData<dim, PatchData>& ld, std::vector<ResourcePair> const& pairs)
{
    Plan<dim> plan{region};
    std::size_t buffered = 0; // local borders' buffer

    auto const pieces = [&](auto const& res, Patch<dim> const& dst, std::size_t const d,
                            auto const& links, ResourcePair const& pair, auto& out) {
        using T               = box_data_value_t<decltype(res), dim>;
        auto const dst_ghost  = ghost_box(res, dst.box);
        auto const dst_domain = domain_box(res, dst.box);

        for (auto const& link : links)
        {
            auto const src_box = region == Region::Ghosts ? domain_box(res, link.patch.box)
                                                          : ghost_box(res, link.patch.box);
            auto const overlap = dst_ghost * core::shift(src_box, link.shift);
            if (!overlap)
                continue;

            auto const add = [&](auto const& box) {
                auto const bytes = static_cast<std::size_t>(box.size()) * sizeof(T);
                out.push_back({d, link, pair, box, 0, bytes, dst.rank});
                if (region == Region::Borders and d != not_local and link.local != not_local)
                {
                    out.back().offset = buffered;
                    buffered += bytes;
                }
            };
            if (region == Region::Ghosts) // never into dst's own domain
                for (auto const& box : overlap->remove(dst_domain))
                    add(box);
            else
                add(*overlap);
        }
    };

    for (auto const& pair : pairs)
    {
        for (std::size_t d = 0; d < n.local.size(); ++d)
            with_box_data<dim>(ld.data[d], pair.dst, [&](auto const& res) {
                check_reach(n, pair.dst, n.local[d].box, ghost_box(res, n.local[d].box));
                pieces(res, n.local[d], d, n.of[d], pair, plan.pieces);
            });
        if (!n.remote.empty())
            with_box_data<dim>(ld.data[0], pair.dst, [&](auto const& res) {
                for (auto const& [dst, links] : n.remote)
                    pieces(res, dst, not_local, links, pair, plan.outgoing);
            });
    }

    plan.buffer.resize(buffered);
    return plan;
}

template<std::size_t dim>
bool detail::is_local(Piece<dim> const& piece)
{
    return piece.dst != not_local and piece.src.local != not_local;
}

template<std::size_t dim>
void detail::check_local(Plan<dim> const& plan)
{
    auto const throw_for = [](Piece<dim> const& piece) {
        throw std::runtime_error("damn::amr::copy: patch " + std::to_string(piece.src.patch.id)
                                 + " is " + std::string{to_string(piece.src.distance)}
                                 + ", copies between ranks are damn::mpi::copy's");
    };
    for (auto const& piece : plan.pieces)
        if (!is_local(piece))
            throw_for(piece);
    if (!plan.outgoing.empty())
        throw_for(plan.outgoing.front());
}

template<typename T>
T detail::read(std::byte const*& at)
{
    T v;
    std::memcpy(&v, at, sizeof(T));
    at += sizeof(T);
    return v;
}

template<typename T>
void detail::write(std::vector<std::byte>& out, T const& v)
{
    static_assert(std::is_trivially_copyable_v<T>);
    auto const at = out.size();
    out.resize(at + sizeof(T));
    std::memcpy(out.data() + at, &v, sizeof(T));
}

template<std::size_t dim, typename PatchData>
void detail::assign(Plan<dim> const& plan, LevelData<dim, PatchData>& ld)
{
    for (auto const& piece : plan.pieces)
        if (is_local(piece))
            with_box_data<dim>(ld.data[piece.src.local], piece.names.src, [&](auto const& src) {
                with_box_data<dim>(ld.data[piece.dst], piece.names.dst, [&](auto& dst) {
                    auto const& dst_box = ld.patches[piece.dst].box;
                    core::for_each_point(piece.box, [&](auto const& p) {
                        value(dst, dst_box, p)
                            = value(src, piece.src.patch.box, p - piece.src.shift);
                    });
                });
            });
}

template<std::size_t dim, typename PatchData>
void detail::pack_local(Plan<dim>& plan, LevelData<dim, PatchData>& ld)
{
    for (auto const& piece : plan.pieces)
        if (is_local(piece))
            with_box_data<dim>(ld.data[piece.src.local], piece.names.src, [&](auto const& src) {
                using T  = box_data_value_t<decltype(src), dim>;
                auto* at = plan.buffer.data() + piece.offset;
                core::for_each_point(piece.box, [&](auto const& p) {
                    T const v{value(src, piece.src.patch.box, p - piece.src.shift)};
                    std::memcpy(at, &v, sizeof(T));
                    at += sizeof(T);
                });
            });
}

template<Arrival arrival, std::size_t dim, typename PatchData>
void detail::unpack_local(Plan<dim>& plan, LevelData<dim, PatchData>& ld)
{
    for (auto const& piece : plan.pieces)
        if (is_local(piece))
        {
            std::byte const* at = plan.buffer.data() + piece.offset;
            unpack_piece<arrival>(plan.region, piece, ld, at);
        }
}

template<std::size_t dim, typename PatchData>
void detail::append_items(Plan<dim> const& plan, LevelData<dim, PatchData>& ld)
{
    for (auto const& piece : plan.pieces)
        if (is_local(piece))
            with_particle_data<dim>(
                ld.data[piece.src.local], piece.names.src, [&](auto const& src) {
                    with_particle_data<dim>(ld.data[piece.dst], piece.names.dst, [&](auto& dst) {
                        using Item         = typename std::decay_t<decltype(dst)>::value_type;
                        auto const leaving = [&](auto const& item) {
                            return isIn(cell(item) + piece.src.shift, piece.box);
                        };

                        if (static_cast<void const*>(&src) != static_cast<void const*>(&dst))
                        {
                            for_each_item(src, [&](auto const& item) {
                                if (leaving(item))
                                    append(dst, item, piece.src.shift);
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
                            append(dst, item, piece.src.shift);
                    });
                });
}


// ---------------------------------------------------------------- pieces as bytes

template<std::size_t dim, typename PatchData>
void pack_piece(Region const region, Piece<dim> const& piece, LevelData<dim, PatchData>& ld,
                std::vector<std::byte>& out)
{
    auto& src_data = ld.data[piece.src.local];
    if (region == Region::Leaving)
    {
        detail::with_particle_data<dim>(src_data, piece.names.src, [&](auto const& src) {
            auto const count_at = out.size();
            detail::write(out, std::uint64_t{0});
            std::uint64_t count = 0;
            for_each_item(src, [&](auto const& item) {
                if (isIn(cell(item) + piece.src.shift, piece.box))
                {
                    detail::write(out, item);
                    ++count;
                }
            });
            std::memcpy(out.data() + count_at, &count, sizeof(count));
        });
        return;
    }
    detail::with_box_data<dim>(src_data, piece.names.src, [&](auto const& src) {
        using T = box_data_value_t<decltype(src), dim>;
        out.reserve(out.size() + piece.bytes);
        core::for_each_point(piece.box, [&](auto const& p) {
            detail::write(out, T{value(src, piece.src.patch.box, p - piece.src.shift)});
        });
    });
}

template<Arrival arrival, std::size_t dim, typename PatchData>
void unpack_piece(Region const region, Piece<dim> const& piece, LevelData<dim, PatchData>& ld,
                  std::byte const*& at)
{
    auto& dst_data = ld.data[piece.dst];
    if (region == Region::Leaving)
    {
        detail::with_particle_data<dim>(dst_data, piece.names.dst, [&](auto& dst) {
            using Item       = typename std::decay_t<decltype(dst)>::value_type;
            auto const count = detail::read<std::uint64_t>(at);
            for (std::uint64_t i = 0; i < count; ++i)
                append(dst, detail::read<Item>(at), piece.src.shift);
        });
        return;
    }
    detail::with_box_data<dim>(dst_data, piece.names.dst, [&](auto& dst) {
        using T             = box_data_value_t<decltype(dst), dim>;
        auto const& dst_box = ld.patches[piece.dst].box;
        core::for_each_point(piece.box, [&](auto const& p) {
            auto const v = detail::read<T>(at);
            auto& d      = value(dst, dst_box, p);
            if constexpr (arrival == Arrival::Sum)
                d += v;
            else if constexpr (arrival == Arrival::Max)
                d = std::max(d, v);
            else
                d = v;
        });
    });
}

} // namespace damn::amr

#endif // DAMN_AMR_COPY_HPP
