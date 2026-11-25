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
//                                       with the sum of all patches' values (deposits), the
//                                       same bits on every patch (see fold_sums)
//     max_borders     Borders, Max      as sum_borders, with the max
//     gather_leaving  Leaving, Append   (CellItemData) items a neighbour holds outside its
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
#include "damn/core/logger.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/memory/scratch.hpp"
#include "damn/amr/box_data.hpp"
#include "damn/amr/cell_item_data.hpp"
#include "damn/amr/cell_items.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/topology.hpp"
#include "damn/core/def/damn_config.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <string_view>
#include <type_traits>
#include <vector>
#include <iterator>
#include <map>
#include <concepts>

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

// a copy's name, for scopes
template<Region region, Arrival arrival>
MKN_KUL_FN_SIG_RET constexpr std::string_view name(CopyOp<region, arrival>)
{
    if constexpr (region == Region::Ghosts)
        return "fill_ghosts";
    else if constexpr (region == Region::Leaving)
        return "gather_leaving";
    else if constexpr (arrival == Arrival::Sum)
        return "sum_borders";
    else
        return "max_borders";
}


// how a copy reaches a piece's resources on a patch's data:
//   by_name     looked up by their names (for_each_resource), the default
//   accessors   the caller's, one per pair in the pairs' order (e.g. a std::vector):
//               access[piece.pair].src(patch_data) and .dst(patch_data) return the typed
//               resources directly (a reference, or a handle holding a pointer), no lookup;
//               names still plan (the geometry, the plan's key)
struct ByName
{
};
inline constexpr ByName by_name{};

namespace detail
{
    // fn(resource) for the resource called name, BoxData or CellItemData
    template<std::size_t dim, bool items, typename PatchData, typename Fn>
    void with_resource_of_kind(PatchData& patch_data, std::string const& name, Fn&& fn);

    template<std::size_t dim, typename PatchData, typename Fn>
    void with_box_data(PatchData& patch_data, std::string const& name, Fn&& fn);

    template<std::size_t dim, typename PatchData, typename Fn>
    void with_cell_item_data(PatchData& patch_data, std::string const& name, Fn&& fn);

    template<std::size_t dim>
    MKN_KUL_FN_SIG_RET bool contains(core::Box<int, dim> const& outer,
                                     core::Box<int, dim> const& inner);

    // throws if a resource's ghost box reaches past the neighbours' reach
    template<std::size_t dim>
    void check_reach(Neighbours<dim> const& n, std::string const& name,
                     core::Box<int, dim> const& patch_box,
                     core::Box<int, dim> const& resource_ghost_box);

    // the pieces of leaving items (Region::Leaving): CellItemData
    template<std::size_t dim, typename PatchData>
    MKN_KUL_FN_SIG_RET Plan<dim> item_plan(Neighbours<dim> const& n, LevelData<dim, PatchData>& ld,
                                           std::vector<ResourcePair> const& pairs);

    // the pieces of ghosts or borders: BoxData; local borders through the plan's buffer
    template<std::size_t dim, typename PatchData>
    MKN_KUL_FN_SIG_RET Plan<dim> box_plan(Region region, Neighbours<dim> const& n,
                                          LevelData<dim, PatchData>& ld,
                                          std::vector<ResourcePair> const& pairs);

    template<std::size_t dim>
    MKN_KUL_FN_SIG_RET bool is_local(Piece<dim> const& piece);

    // throws if the plan has pieces to or from another rank
    template<std::size_t dim>
    void check_local(Plan<dim> const& plan);

    template<typename T>
    MKN_KUL_FN_SIG_RET T read(std::byte const*& at);

    template<typename T>
    void write(std::vector<std::byte>& out, T const& v);

    // a BoxData resource whose values are a C ordered array over a box: box_span(resource,
    //  patch box) its data and the box of its points. Its pieces are copied span by span
    //  (core::for_each_span: a run along the last direction, contiguous in memory) instead of
    //  point by point, in the same point order: the same bytes, the same sums
    template<typename R, std::size_t dim>
    concept SpanBoxData
        = BoxData<std::remove_cvref_t<R>, dim> and requires(R& r, core::Box<int, dim> const& box) {
              { box_span(r, box).box } -> std::convertible_to<core::Box<int, dim>>;
              requires std::same_as<
                  std::remove_cv_t<std::remove_pointer_t<decltype(box_span(r, box).ptr)>>,
                  box_data_value_t<std::remove_cvref_t<R>, dim>>;
          };

    // the first value of the span starting at p (amr index space) of a resource's box_span
    template<std::size_t dim>
    MKN_KUL_FN_SIG_RET auto* row(auto const& span, core::Point<int, dim> const& p)
    {
        auto const shape = span.box.shape();
        std::size_t flat = 0;
        for (std::size_t d = 0; d < dim; ++d)
            flat = flat * static_cast<std::size_t>(shape[d])
                   + static_cast<std::size_t>(p[d] - span.box.lower[d]);
        return span.ptr + flat;
    }

    // local pieces only, below: pieces from or to other ranks are damn::mpi::copy's
    // fn(a piece's source / destination resource on patch_data), as access reaches it
    template<std::size_t dim, bool items, typename PatchData, typename Fn>
    void src_of(ByName, PatchData& patch_data, Piece<dim> const& piece, Fn&& fn);
    template<std::size_t dim, bool items, typename PatchData, typename Fn>
    void dst_of(ByName, PatchData& patch_data, Piece<dim> const& piece, Fn&& fn);
    template<std::size_t dim, bool items, typename Access, typename PatchData, typename Fn>
        requires(!std::is_same_v<Access, ByName>)
    void src_of(Access const& access, PatchData& patch_data, Piece<dim> const& piece, Fn&& fn);
    template<std::size_t dim, bool items, typename Access, typename PatchData, typename Fn>
        requires(!std::is_same_v<Access, ByName>)
    void dst_of(Access const& access, PatchData& patch_data, Piece<dim> const& piece, Fn&& fn);

    template<std::size_t dim, typename Access, typename PatchData>
    void assign(Plan<dim> const& plan, Access const& access, LevelData<dim, PatchData>& ld);

    // the copy done on the device (device_copy.hpp: GPU builds, every piece's resources on the
    //  device, Assign, Sum or Append); false: the host path's to do
    template<Arrival arrival, std::size_t dim, typename Access, typename PatchData>
    bool on_device([[maybe_unused]] Plan<dim>& plan, [[maybe_unused]] Access const& access,
                   [[maybe_unused]] LevelData<dim, PatchData>& ld) // used by GPU builds only
    {
#if DAMN_HAVE_GPU
        if constexpr (arrival == Arrival::Assign)
            return assign_on_device(plan, access, ld);
        else if constexpr (arrival == Arrival::Sum)
            return sum_on_device(plan, access, ld);
        else if constexpr (arrival == Arrival::Append)
            return append_on_device(plan, access, ld);
#endif
        return false;
    }

    template<std::size_t dim, typename Access, typename PatchData>
    void append_items(Plan<dim> const& plan, Access const& access, LevelData<dim, PatchData>& ld);

    // after a copy appended to them: each destination resource once (cell_items::appended)
    template<std::size_t dim, typename Access, typename PatchData>
    void appended(Plan<dim> const& plan, Access const& access, LevelData<dim, PatchData>& ld);

    // sums and maxes: every local source read into the buffer, then (unpack) written
    template<std::size_t dim, typename Access, typename PatchData>
    void pack_local(Plan<dim>& plan, Access const& access, LevelData<dim, PatchData>& ld);

    template<Arrival arrival, std::size_t dim, typename Access, typename PatchData>
    void unpack_local(Plan<dim>& plan, Access const& access, LevelData<dim, PatchData>& ld);

    // what follows from a new plan's pieces alone (Plan's from, fold_order, fold_groups)
    template<std::size_t dim>
    void finish_plan(Plan<dim>& plan);

    // plan.sources: where each piece's packed values are (by its index in plan.pieces): local
    //  ones in the plan's buffer, other ranks' null (set by damn::mpi::copy)
    template<std::size_t dim>
    void local_sources(Plan<dim>& plan);

    struct FoldOwn; // fold_sums' scratch (core::scratch tag)

    // the sums, in the same order on every patch holding a point (PHAREHUB/PHARE#1581): per
    //  destination and resource, the summed points are zeroed, then the contributions added in
    //  the lexicographic order of their patch's box in the destination's space (its image), the
    //  destination's own included. Every holder of a point has the same contributing patches,
    //  their images one shift apart, and the order is translation invariant: the same values
    //  added in the same order, the same bits ((a + b) + c and (a + c) + b can differ)
    template<std::size_t dim, typename Access, typename PatchData>
    void fold_sums(Plan<dim> const& plan, Access const& access, LevelData<dim, PatchData>& ld);
} // namespace detail


// a piece's data from its (local) source, appended to out: field values in the box's point
// order, or the number of items then the items
template<std::size_t dim, typename Access, typename PatchData>
void pack_piece(Region region, Piece<dim> const& piece, Access const& access,
                LevelData<dim, PatchData>& ld, std::vector<std::byte>& out);

// a piece's data, packed by pack_piece on another rank, into its (local) destination; at
// moves past it
template<Arrival arrival, std::size_t dim, typename Access, typename PatchData>
void unpack_piece(Region region, Piece<dim> const& piece, Access const& access,
                  LevelData<dim, PatchData>& ld, std::byte const*& at);


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
    {
        DAMN_LOG_SCOPE(2, "damn::amr::plan_for"); // once per region and resources
        found = n.plans
                    .emplace(std::move(key), region == Region::Leaving
                                                 ? detail::item_plan(n, ld, pairs)
                                                 : detail::box_plan(region, n, ld, pairs))
                    .first;
        detail::finish_plan(found->second);
    }
    return found->second;
}

// the number of accessors is the number of pairs (by_name: nothing to check)
template<typename Access>
void check_access(Access const& access, std::size_t const pairs)
{
    if constexpr (!std::is_same_v<Access, ByName>)
        if (std::size(access) != pairs)
            throw std::runtime_error("damn::amr::copy: one accessor per resource pair");
}

// on one rank: throws if any piece comes from, or must go to, another rank. access: how the
//  pieces' resources are reached (by_name, or accessors one per pair)
template<Region region, Arrival arrival, std::size_t dim, typename Access, typename PatchData>
void copy(CopyOp<region, arrival> const op, Neighbours<dim>& n,
          std::vector<ResourcePair> const& pairs, Access const& access,
          LevelData<dim, PatchData>& ld)
{
    DAMN_LOG_SCOPE(2, "damn::amr::copy<" + std::string{name(op)} + ">");
    check_access(access, pairs.size());
    auto& plan = plan_for(region, n, ld, pairs);
    detail::check_local(plan);

    if constexpr (arrival == Arrival::Assign)
    {
        if (!detail::on_device<arrival>(plan, access, ld))
            detail::assign(plan, access, ld);
    }
    else if constexpr (arrival == Arrival::Append)
    {
        if (!detail::on_device<arrival>(plan, access, ld))
            detail::append_items(plan, access, ld);
        detail::appended(plan, access, ld);
    }
    else if (!detail::on_device<arrival>(plan, access, ld))
    {
        detail::pack_local(plan, access, ld);
        if constexpr (arrival == Arrival::Sum)
        {
            detail::local_sources(plan);
            detail::fold_sums(plan, access, ld);
        }
        else
            detail::unpack_local<arrival>(plan, access, ld);
    }
}

template<Region region, Arrival arrival, std::size_t dim, typename PatchData>
void copy(CopyOp<region, arrival> const op, Neighbours<dim>& n,
          std::vector<ResourcePair> const& pairs, LevelData<dim, PatchData>& ld)
{
    copy(op, n, pairs, by_name, ld);
}

// fields: the same name on both sides
template<Region region, Arrival arrival, std::size_t dim, typename PatchData>
void copy(CopyOp<region, arrival> const op, Neighbours<dim>& n,
          std::vector<std::string> const& names, LevelData<dim, PatchData>& ld)
{
    copy(op, n, same_names(names), ld);
}


// ---------------------------------------------------------------- detail

template<std::size_t dim, bool items, typename PatchData, typename Fn>
void detail::with_resource_of_kind(PatchData& patch_data, std::string const& name, Fn&& fn)
{
    bool const found = with_resource(patch_data, name, [&](auto& resource) {
        using R = std::decay_t<decltype(resource)>;
        if constexpr (items)
        {
            if constexpr (CellItemData<R, dim>)
                fn(resource);
            else
                throw std::runtime_error("damn::amr::copy: " + name + " is not CellItemData");
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
void detail::with_cell_item_data(PatchData& patch_data, std::string const& name, Fn&& fn)
{
    with_resource_of_kind<dim, true>(patch_data, name, fn);
}

template<std::size_t dim, bool items, typename PatchData, typename Fn>
void detail::src_of(ByName, PatchData& patch_data, Piece<dim> const& piece, Fn&& fn)
{
    with_resource_of_kind<dim, items>(patch_data, piece.names.src, fn);
}

template<std::size_t dim, bool items, typename PatchData, typename Fn>
void detail::dst_of(ByName, PatchData& patch_data, Piece<dim> const& piece, Fn&& fn)
{
    with_resource_of_kind<dim, items>(patch_data, piece.names.dst, fn);
}

template<std::size_t dim, bool items, typename Access, typename PatchData, typename Fn>
    requires(!std::is_same_v<Access, ByName>)
void detail::src_of(Access const& access, PatchData& patch_data, Piece<dim> const& piece, Fn&& fn)
{
    decltype(auto) resource = access[piece.pair].src(patch_data); // a reference or a handle
    using R                 = std::decay_t<decltype(resource)>;
    // as by name: a resource of the other kind throws (the region is a run time choice in
    //  pack_piece / unpack_piece, both kinds' code is compiled)
    if constexpr (items ? CellItemData<R, dim> : BoxData<R, dim>)
        fn(resource);
    else
        throw std::runtime_error(std::string{"damn::amr::copy: an accessor's resource is not "}
                                 + (items ? "CellItemData" : "BoxData"));
}

template<std::size_t dim, bool items, typename Access, typename PatchData, typename Fn>
    requires(!std::is_same_v<Access, ByName>)
void detail::dst_of(Access const& access, PatchData& patch_data, Piece<dim> const& piece, Fn&& fn)
{
    decltype(auto) resource = access[piece.pair].dst(patch_data); // a reference or a handle
    using R                 = std::decay_t<decltype(resource)>;
    // as by name: a resource of the other kind throws (the region is a run time choice in
    //  pack_piece / unpack_piece, both kinds' code is compiled)
    if constexpr (items ? CellItemData<R, dim> : BoxData<R, dim>)
        fn(resource);
    else
        throw std::runtime_error(std::string{"damn::amr::copy: an accessor's resource is not "}
                                 + (items ? "CellItemData" : "BoxData"));
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
Plan<dim> detail::item_plan(Neighbours<dim> const& n, LevelData<dim, PatchData>& ld,
                            std::vector<ResourcePair> const& pairs)
{
    Plan<dim> plan{Region::Leaving};
    auto const pieces = [&](auto const& res, Patch<dim> const& dst, std::size_t const d,
                            auto const& links, ResourcePair const& pair, std::uint32_t const k,
                            auto& out) {
        auto const dst_domain = domain_box(res, dst.box);
        for (auto const& link : links)
            if (auto const overlap
                = dst_domain * core::shift(ghost_box(res, link.patch.box), link.shift))
                out.push_back({d, link, pair, *overlap, 0, 0, dst.rank, k});
    };

    for (std::uint32_t k = 0; k < pairs.size(); ++k)
    {
        auto const& pair = pairs[k];
        for (std::size_t d = 0; d < n.local.size(); ++d)
            with_cell_item_data<dim>(ld.data[d], pair.dst, [&](auto const& res) {
                check_reach(n, pair.dst, n.local[d].box, ghost_box(res, n.local[d].box));
                pieces(res, n.local[d], d, n.of[d], pair, k, plan.pieces);
            });
        if (!n.remote.empty())
            with_cell_item_data<dim>(ld.data[0], pair.dst, [&](auto const& res) {
                for (auto const& [dst, links] : n.remote)
                    pieces(res, dst, not_local, links, pair, k, plan.outgoing);
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
                            auto const& links, ResourcePair const& pair, std::uint32_t const k,
                            auto& out) {
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
                out.push_back({d, link, pair, box, 0, bytes, dst.rank, k});
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

    for (std::uint32_t k = 0; k < pairs.size(); ++k)
    {
        auto const& pair = pairs[k];
        for (std::size_t d = 0; d < n.local.size(); ++d)
            with_box_data<dim>(ld.data[d], pair.dst, [&](auto const& res) {
                check_reach(n, pair.dst, n.local[d].box, ghost_box(res, n.local[d].box));
                pieces(res, n.local[d], d, n.of[d], pair, k, plan.pieces);
            });
        if (!n.remote.empty())
            with_box_data<dim>(ld.data[0], pair.dst, [&](auto const& res) {
                for (auto const& [dst, links] : n.remote)
                    pieces(res, dst, not_local, links, pair, k, plan.outgoing);
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

template<std::size_t dim, typename Access, typename PatchData>
void detail::assign(Plan<dim> const& plan, Access const& access, LevelData<dim, PatchData>& ld)
{
    for (auto const& piece : plan.pieces)
        if (is_local(piece))
            src_of<dim, false>(access, ld.data[piece.src.local], piece, [&](auto const& src) {
                auto const& dst_box = ld.patches[piece.dst].box;
                auto const points   = [&](auto& dst) {
                    core::for_each_point(piece.box, [&](auto const& p) {
                        value(dst, dst_box, p)
                            = value(src, piece.src.patch.box, p - piece.src.shift);
                    });
                };
                // each side's kind tested where its type is known: an expression of src alone
                //  under a test of dst is not dependent once src's lambda is instantiated, and
                //  clang looks it up then
                if constexpr (SpanBoxData<decltype(src), dim>)
                {
                    auto const s = box_span(src, piece.src.patch.box);
                    dst_of<dim, false>(access, ld.data[piece.dst], piece, [&](auto& dst) {
                        if constexpr (SpanBoxData<decltype(dst), dim>)
                        {
                            using T = box_data_value_t<std::remove_cvref_t<decltype(dst)>, dim>;
                            auto const d = box_span(dst, dst_box);
                            core::for_each_span(
                                piece.box, [&](auto const& p, std::uint32_t const n) {
                                    std::memcpy(row<dim>(d, p), row<dim>(s, p - piece.src.shift),
                                                n * sizeof(T));
                                });
                        }
                        else
                            points(dst);
                    });
                }
                else
                    dst_of<dim, false>(access, ld.data[piece.dst], piece, points);
            });
}

template<std::size_t dim, typename Access, typename PatchData>
void detail::pack_local(Plan<dim>& plan, Access const& access, LevelData<dim, PatchData>& ld)
{
    for (auto const& piece : plan.pieces)
        if (is_local(piece))
            src_of<dim, false>(access, ld.data[piece.src.local], piece, [&](auto const& src) {
                using T  = box_data_value_t<std::remove_cvref_t<decltype(src)>, dim>;
                auto* at = plan.buffer.data() + piece.offset;
                if constexpr (SpanBoxData<decltype(src), dim>)
                {
                    auto const s = box_span(src, piece.src.patch.box);
                    core::for_each_span(piece.box, [&](auto const& p, std::uint32_t const n) {
                        std::memcpy(at, row<dim>(s, p - piece.src.shift), n * sizeof(T));
                        at += n * sizeof(T);
                    });
                }
                else
                    core::for_each_point(piece.box, [&](auto const& p) {
                        T const v{value(src, piece.src.patch.box, p - piece.src.shift)};
                        std::memcpy(at, &v, sizeof(T));
                        at += sizeof(T);
                    });
            });
}

template<Arrival arrival, std::size_t dim, typename Access, typename PatchData>
void detail::unpack_local(Plan<dim>& plan, Access const& access, LevelData<dim, PatchData>& ld)
{
    for (auto const& piece : plan.pieces)
        if (is_local(piece))
        {
            std::byte const* at = plan.buffer.data() + piece.offset;
            unpack_piece<arrival>(plan.region, piece, access, ld, at);
        }
}

template<std::size_t dim>
void detail::finish_plan(Plan<dim>& plan)
{
    auto const& pieces = plan.pieces;

    std::map<int, std::vector<std::size_t>> from;
    for (std::size_t i = 0; i < pieces.size(); ++i)
        if (pieces[i].src.local == not_local)
            from[pieces[i].src.patch.rank].push_back(i);
    plan.from.assign(from.begin(), from.end());

    if (plan.region != Region::Borders)
        return;
    auto const image = [&](std::size_t const i) {
        return core::shift(pieces[i].src.patch.box, pieces[i].src.shift);
    };
    auto& order = plan.fold_order;
    order.resize(pieces.size());
    for (std::size_t i = 0; i < order.size(); ++i)
        order[i] = i;
    std::sort(order.begin(), order.end(), [&](auto const i, auto const j) {
        if (pieces[i].dst != pieces[j].dst)
            return pieces[i].dst < pieces[j].dst;
        if (pieces[i].names != pieces[j].names)
            return pieces[i].names < pieces[j].names;
        return core::lexicographic_less(image(i), image(j));
    });
    for (std::size_t begin = 0; begin < order.size();)
    {
        auto& group = plan.fold_groups.emplace_back();
        group.begin = begin;
        group.end   = begin + 1;
        while (group.end < order.size() and pieces[order[group.end]].dst == pieces[order[begin]].dst
               and pieces[order[group.end]].names == pieces[order[begin]].names)
            ++group.end;
        for (auto k = group.begin; k < group.end; ++k) // each point once: what earlier pieces miss
        {
            std::vector<core::Box<int, dim>> parts{pieces[order[k]].box};
            for (auto const& covered : group.covered)
            {
                std::vector<core::Box<int, dim>> rest;
                for (auto const& part : parts)
                    for (auto const& r : core::minus_ordered(part, covered))
                        rest.push_back(r);
                parts = std::move(rest);
            }
            for (auto const& part : parts)
            {
                group.covered.push_back(part);
                group.points += static_cast<std::size_t>(part.size());
            }
        }
        begin = group.end;
    }
}

template<std::size_t dim>
void detail::local_sources(Plan<dim>& plan)
{
    plan.sources.assign(plan.pieces.size(), nullptr);
    for (std::size_t i = 0; i < plan.pieces.size(); ++i)
        if (is_local(plan.pieces[i]))
            plan.sources[i] = plan.buffer.data() + plan.pieces[i].offset;
}

template<std::size_t dim, typename Access, typename PatchData>
void detail::fold_sums(Plan<dim> const& plan, Access const& access, LevelData<dim, PatchData>& ld)
{
    auto const& pieces = plan.pieces;
    auto const& order  = plan.fold_order;
    auto const image   = [&](std::size_t const i) {
        return core::shift(pieces[i].src.patch.box, pieces[i].src.shift);
    };

    for (auto const& group : plan.fold_groups)
    {
        auto const& first   = pieces[order[group.begin]];
        auto const& dst_box = ld.patches[first.dst].box;
        dst_of<dim, false>(access, ld.data[first.dst], first, [&](auto& dst) {
            using T = box_data_value_t<std::remove_cvref_t<decltype(dst)>, dim>;

            // fn(values, n) span by span, or point by point: the same operations, same order
            auto const each = [&](auto const& box, auto&& fn) {
                if constexpr (SpanBoxData<decltype(dst), dim>)
                {
                    auto const d = box_span(dst, dst_box);
                    core::for_each_span(
                        box, [&](auto const& p, std::uint32_t const n) { fn(row<dim>(d, p), n); });
                }
                else
                    core::for_each_point(
                        box, [&](auto const& p) { fn(&value(dst, dst_box, p), std::uint32_t{1}); });
            };

            // the summed points: the destination's own values kept, then zeroed
            auto const own = core::scratch<T, FoldOwn>(group.points);
            std::size_t o  = 0;
            for (auto const& box : group.covered)
                each(box, [&](T* const v, std::uint32_t const n) {
                    std::copy_n(v, n, own.data() + o);
                    std::fill_n(v, n, T{0});
                    o += n;
                });
            auto const add_own = [&]() {
                std::size_t at = 0;
                for (auto const& box : group.covered)
                    each(box, [&](T* const v, std::uint32_t const n) {
                        for (std::uint32_t j = 0; j < n; ++j)
                            v[j] += own[at + j];
                        at += n;
                    });
            };

            // the contributions in image order, the destination's own at its place
            bool own_added = false;
            for (auto k = group.begin; k < group.end; ++k)
            {
                auto const i = order[k];
                if (!own_added and core::lexicographic_less(dst_box, image(i)))
                {
                    add_own();
                    own_added = true;
                }
                if (!plan.sources[i])
                    throw std::runtime_error("damn::amr::copy: a border sum piece not received");
                std::byte const* at = plan.sources[i];
                each(pieces[i].box, [&](T* const v, std::uint32_t const n) {
                    for (std::uint32_t j = 0; j < n; ++j)
                        v[j] += read<T>(at);
                });
            }
            if (!own_added)
                add_own();
        });
    }
}

template<std::size_t dim, typename Access, typename PatchData>
void detail::append_items(Plan<dim> const& plan, Access const& access,
                          LevelData<dim, PatchData>& ld)
{
    for (auto const& piece : plan.pieces)
        if (is_local(piece))
            src_of<dim, true>(access, ld.data[piece.src.local], piece, [&](auto const& src) {
                dst_of<dim, true>(access, ld.data[piece.dst], piece, [&](auto& dst) {
                    cell_items::append_from(dst, src, piece.box, piece.src.shift);
                });
            });
}

template<std::size_t dim, typename Access, typename PatchData>
void detail::appended(Plan<dim> const& plan, Access const& access, LevelData<dim, PatchData>& ld)
{
    std::vector<std::pair<std::size_t, std::string const*>> done;
    for (auto const& piece : plan.pieces)
    {
        auto const key = std::pair{piece.dst, &piece.names.dst};
        if (std::any_of(done.begin(), done.end(), [&](auto const& d) {
                return d.first == key.first and *d.second == *key.second;
            }))
            continue;
        done.push_back(key);
        dst_of<dim, true>(access, ld.data[piece.dst], piece,
                          [&](auto& dst) { cell_items::appended(dst); });
    }
}


// ---------------------------------------------------------------- pieces as bytes

template<std::size_t dim, typename Access, typename PatchData>
void pack_piece(Region const region, Piece<dim> const& piece, Access const& access,
                LevelData<dim, PatchData>& ld, std::vector<std::byte>& out)
{
    auto& src_data = ld.data[piece.src.local];
    if (region == Region::Leaving)
    {
        detail::src_of<dim, true>(access, src_data, piece, [&](auto const& src) {
            cell_items::pack(src, piece.box, piece.src.shift, out);
        });
        return;
    }
    detail::src_of<dim, false>(access, src_data, piece, [&](auto const& src) {
        using T = box_data_value_t<std::remove_cvref_t<decltype(src)>, dim>;
        if constexpr (detail::SpanBoxData<decltype(src), dim>)
        {
            auto const s = box_span(src, piece.src.patch.box);
            auto at      = out.size();
            out.resize(at + static_cast<std::size_t>(piece.box.size()) * sizeof(T));
            core::for_each_span(piece.box, [&](auto const& p, std::uint32_t const n) {
                std::memcpy(out.data() + at, detail::row<dim>(s, p - piece.src.shift),
                            n * sizeof(T));
                at += n * sizeof(T);
            });
        }
        else
        {
            out.reserve(out.size() + piece.bytes);
            core::for_each_point(piece.box, [&](auto const& p) {
                detail::write(out, T{value(src, piece.src.patch.box, p - piece.src.shift)});
            });
        }
    });
}

template<Arrival arrival, std::size_t dim, typename Access, typename PatchData>
void unpack_piece(Region const region, Piece<dim> const& piece, Access const& access,
                  LevelData<dim, PatchData>& ld, std::byte const*& at)
{
    auto& dst_data = ld.data[piece.dst];
    if (region == Region::Leaving)
    {
        detail::dst_of<dim, true>(access, dst_data, piece,
                                  [&](auto& dst) { cell_items::unpack(dst, at, piece.src.shift); });
        return;
    }
    detail::dst_of<dim, false>(access, dst_data, piece, [&](auto& dst) {
        using T             = box_data_value_t<std::remove_cvref_t<decltype(dst)>, dim>;
        auto const& dst_box = ld.patches[piece.dst].box;
        if constexpr (detail::SpanBoxData<decltype(dst), dim>)
        {
            auto const d = box_span(dst, dst_box);
            core::for_each_span(piece.box, [&](auto const& p, std::uint32_t const n) {
                T* const out = detail::row<dim>(d, p);
                if constexpr (arrival == Arrival::Assign)
                    std::memcpy(out, at, n * sizeof(T));
                else
                    for (std::uint32_t i = 0; i < n; ++i)
                    {
                        T v;
                        std::memcpy(&v, at + i * sizeof(T), sizeof(T));
                        if constexpr (arrival == Arrival::Sum)
                            out[i] += v;
                        else
                            out[i] = std::max(out[i], v);
                    }
                at += n * sizeof(T);
            });
            return;
        }
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

// after the detail functions it uses (src_of, dst_of, is_local)
#include "damn/amr/device_copy.hpp" // IWYU pragma: export

#endif // DAMN_AMR_COPY_HPP
