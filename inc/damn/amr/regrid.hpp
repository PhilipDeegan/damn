#ifndef DAMN_AMR_REGRID_HPP
#define DAMN_AMR_REGRID_HPP

// a level's data moved to another decomposition of the same cells: doc/amr/regrid.md
// the old data is consumed, each old patch freed once done: the peak is about the old level

#include "damn/core/def.hpp"
#include "damn/core/logger.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/amr/box_data.hpp"
#include "damn/amr/copy.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/level_data.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/cell_item_data.hpp"
#include "damn/amr/cell_items.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace damn::amr
{
// resources whose old ghosts are valid, moved with the data; reach: their widest ghost width
struct RegridGhosts
{
    int reach = 0;
    std::vector<std::string> fields{};    // BoxData resources
    std::vector<std::string> items{};     // CellItemData resources

    MKN_KUL_FN_SIG_RET bool field(std::string_view const name) const
    {
        return std::find(fields.begin(), fields.end(), name) != fields.end();
    }
    MKN_KUL_FN_SIG_RET bool item(std::string_view const name) const
    {
        return std::find(items.begin(), items.end(), name) != items.end();
    }
};

// what of one old patch (or a periodic image of it) goes to one new patch
template<std::size_t dim>
struct Transfer
{
    Patch<dim> from{}, to{};
    core::Point<int, dim> shift{};              // the old patch's image is at from.box + shift
    std::optional<core::Box<int, dim>> cells{}; // (from.box + shift) * to.box, if they share any
    std::size_t src = not_local;                // old LevelData index, if this rank holds from
    std::size_t dst = not_local;                // new LevelData index, if this rank holds to
    // RegridGhosts: their domains hold the new patch's ghost nodes rather than an old ghost
    std::vector<core::Box<int, dim>> near_old{};
};

// both ranks of a pair list the transfers between them in the same order (old level order,
// then new level order): messages need no description of their content
template<std::size_t dim>
struct RegridMap
{
    int level = 0;
    int rank  = 0;
    std::vector<Patch<dim>> old_patches{}, new_patches{}; // the whole levels
    std::vector<Patch<dim>> old_local{}, new_local{};     // this rank's, in LevelData order
    std::vector<std::size_t> kept{}; // kept[new index]: the old index whose data it takes as is
    std::vector<Transfer<dim>> transfers{}; // from or to this rank, local or not
    RegridGhosts ghosts{};
};


// shifts: periodic_shifts, the domain itself excluded
template<std::size_t dim>
MKN_KUL_FN_SIG_RET RegridMap<dim>
make_regrid_map(Level<dim> const& old_level, Level<dim> const& new_level, int const rank,
                std::vector<core::Point<int, dim>> shifts = {}, RegridGhosts ghosts = {})
{
    DAMN_LOG_SCOPE(2, "damn::amr::make_regrid_map");
    if (old_level.number != new_level.number)
        throw std::runtime_error("damn::amr::make_regrid_map: not the same level");

    RegridMap<dim> map{old_level.number,
                       rank,
                       old_level.patches,
                       new_level.patches,
                       local_patches(old_level, rank),
                       local_patches(new_level, rank),
                       {},
                       {},
                       std::move(ghosts)};
    auto const reach = map.ghosts.reach;

    auto const index_in = [](auto const& patches, std::uint32_t const id) {
        for (std::size_t i = 0; i < patches.size(); ++i)
            if (patches[i].id == id)
                return i;
        return not_local;
    };

    // kept: same box, same owner. Every rank finds the same ones
    std::vector<bool> is_kept(new_level.patches.size(), false);
    for (std::size_t n = 0; n < new_level.patches.size(); ++n)
        for (auto const& o : old_level.patches)
            if (o.box == new_level.patches[n].box and o.rank == new_level.patches[n].rank)
                is_kept[n] = true;

    map.kept.assign(map.new_local.size(), not_local);
    for (std::size_t d = 0; d < map.new_local.size(); ++d)
        for (std::size_t s = 0; s < map.old_local.size(); ++s)
            if (map.old_local[s].box == map.new_local[d].box)
                map.kept[d] = s;

    // overlapping or touching (a border's primal points), with valid ghosts within reach of
    //  each other: every rank lists the same, in the same order
    shifts.insert(shifts.begin(), core::Point<int, dim>{});
    std::vector<core::Box<int, dim>> images;
    if (reach > 0)
        for (auto const& o : old_level.patches)
            for (auto const& shift : shifts)
                images.push_back(core::shift(o.box, shift));
    for (auto const& o : old_level.patches)
        for (auto const& shift : shifts)
            for (std::size_t n = 0; n < new_level.patches.size(); ++n)
            {
                auto const& to = new_level.patches[n];
                if (is_kept[n] or (o.rank != rank and to.rank != rank))
                    continue;
                auto const image = core::shift(o.box, shift);
                if (!(core::grow(image, reach + 1) * core::grow(to.box, reach)))
                    continue;
                Transfer<dim> t{o,
                                to,
                                shift,
                                image * to.box,
                                index_in(map.old_local, o.id),
                                index_in(map.new_local, to.id)};
                if (reach > 0)
                    for (auto const& other : images)
                        if (core::grow(other, reach + 1) * core::grow(to.box, reach))
                            t.near_old.push_back(other);
                map.transfers.push_back(std::move(t));
            }
    return map;
}


namespace detail
{
    // in an order both sides agree on: the domain points both share; with valid ghosts, the new
    //  ghost nodes the old domain holds, then those the old patch's ghosts hold and no old
    //  domain does
    template<std::size_t dim>
    MKN_KUL_FN_SIG_RET std::vector<core::Box<int, dim>>
    transfer_points(RegridMap<dim> const& map, Transfer<dim> const& t, std::string_view const name,
                    auto const& res)
    {
        auto const from = core::shift(t.from.box, t.shift);
        if (!map.ghosts.field(name))
        {
            if (auto const b = domain_box(res, from) * domain_box(res, t.to.box))
                return {*b};
            return {};
        }
        std::vector<core::Box<int, dim>> out;
        auto const to = ghost_box(res, t.to.box);
        if (auto const b = domain_box(res, from) * to)
            out.push_back(*b);
        if (auto const g = ghost_box(res, from) * to)
        {
            std::vector<core::Box<int, dim>> rest{*g};
            for (auto const& other : t.near_old)
            {
                std::vector<core::Box<int, dim>> left;
                for (auto const& r : rest)
                    for (auto const& piece : core::minus_ordered(r, domain_box(res, other)))
                        left.push_back(piece);
                rest = std::move(left);
            }
            out.insert(out.end(), rest.begin(), rest.end());
        }
        return out;
    }

    // the shared cells, or, its ghosts valid, its ghost cells (own width) on both sides
    template<std::size_t dim>
    MKN_KUL_FN_SIG_RET std::optional<core::Box<int, dim>>
    transfer_cells(RegridMap<dim> const& map, Transfer<dim> const& t, std::string_view const name,
                   auto const& res)
    {
        if (!map.ghosts.item(name))
            return t.cells;
        return ghost_box(res, core::shift(t.from.box, t.shift)) * ghost_box(res, t.to.box);
    }

    template<std::size_t dim, typename PatchData>
    void for_each_regridded(PatchData& data, auto&& fn)
    {
        for_each_resource(data, [&](std::string_view const name, auto& resource) {
            using R = std::decay_t<decltype(resource)>;
            if constexpr (BoxData<R, dim> or CellItemData<R, dim>)
                fn(name, resource);
        });
    }

    // resource by resource: field values in the box's point order, or the number of items then
    //  the items
    template<std::size_t dim, typename PatchData>
    void pack_transfer(RegridMap<dim> const& map, Transfer<dim> const& t, PatchData& src,
                       std::vector<std::byte>& out)
    {
        for_each_regridded<dim>(src, [&](std::string_view const name, auto const& res) {
            using R = std::decay_t<decltype(res)>;
            if constexpr (CellItemData<R, dim>)
            {
                if (auto const cells = transfer_cells(map, t, name, res))
                    cell_items::pack(res, *cells, t.shift, out);
                else // no cells: a zero count, damn's (both sides have the transfer)
                    write(out, std::uint64_t{0});
            }
            else
                for (auto const& box : transfer_points(map, t, name, res))
                {
                    using T = box_data_value_t<R const, dim>;
                    out.reserve(out.size() + static_cast<std::size_t>(box.size()) * sizeof(T));
                    core::for_each_point(box, [&](auto const& p) {
                        write(out, T{value(res, t.from.box, p - t.shift)});
                    });
                }
        });
    }

    // at moves past it
    template<std::size_t dim, typename PatchData>
    void unpack_transfer(RegridMap<dim> const& map, Transfer<dim> const& t, PatchData& dst,
                         std::byte const*& at)
    {
        for_each_regridded<dim>(dst, [&](std::string_view const name, auto& res) {
            using R = std::decay_t<decltype(res)>;
            if constexpr (CellItemData<R, dim>)
            {
                if (transfer_cells(map, t, name, res))
                    cell_items::unpack(res, at, t.shift);
                else
                    (void)read<std::uint64_t>(at);
            }
            else
                for (auto const& box : transfer_points(map, t, name, res))
                {
                    using T = box_data_value_t<R, dim>;
                    core::for_each_point(
                        box, [&](auto const& p) { value(res, t.to.box, p) = read<T>(at); });
                }
        });
    }

    template<std::size_t dim, typename PatchData>
    void copy_transfer(RegridMap<dim> const& map, Transfer<dim> const& t, PatchData& src,
                       PatchData& dst)
    {
        for_each_regridded<dim>(dst, [&](std::string_view const name, auto& d) {
            using R          = std::decay_t<decltype(d)>;
            bool const found = with_resource(src, name, [&](auto const& s) {
                if constexpr (std::is_same_v<std::decay_t<decltype(s)>, R>)
                {
                    if constexpr (CellItemData<R, dim>)
                    {
                        if (auto const cells = transfer_cells(map, t, name, d))
                            cell_items::append_from(d, s, *cells, t.shift);
                    }
                    else
                        for (auto const& box : transfer_points(map, t, name, d))
                            core::for_each_point(box, [&](auto const& p) {
                                value(d, t.to.box, p) = value(s, t.from.box, p - t.shift);
                            });
                }
                else
                    throw std::runtime_error("damn::amr::regrid: " + std::string{name}
                                             + " is not the same kind on both sides");
            });
            if (!found)
                throw std::runtime_error("damn::amr::regrid: no resource " + std::string{name}
                                         + " in the old patch data");
        });
    }

    // send(transfer, old data, new_data) packs the transfers to other ranks' patches;
    //  finish(new_data) receives the rest. new_data(d): new local patch d's data, made on first
    //  use
    template<std::size_t dim, typename PatchData, typename Make, typename Send, typename Finish>
    LevelData<dim, PatchData> regrid(RegridMap<dim> const& map, LevelData<dim, PatchData>&& old,
                                     Make&& make, Send&& send, Finish&& finish)
    {
        DAMN_LOG_SCOPE(1, "damn::amr::regrid");
        if (old.level != map.level or old.patches != map.old_local)
            throw std::runtime_error("damn::amr::regrid: the map is not the level data's");

        std::vector<std::optional<PatchData>> olds;
        olds.reserve(old.data.size());
        for (auto& d : old.data)
            olds.emplace_back(std::move(d));
        old.data.clear(); // not = {}: an initializer_list, copies (move only patch data)

        std::vector<std::optional<PatchData>> news(map.new_local.size());
        auto const make_new = [&](std::size_t const d) -> PatchData& {
            if (!news[d])
                news[d].emplace(make(map.new_local[d]));
            return *news[d];
        };

        for (std::size_t d = 0; d < map.kept.size(); ++d)
            if (map.kept[d] != not_local)
            {
                news[d] = std::move(olds[map.kept[d]]);
                olds[map.kept[d]].reset();
            }

        std::vector<std::vector<Transfer<dim> const*>> from(olds.size());
        for (auto const& t : map.transfers)
            if (t.src != not_local)
                from[t.src].push_back(&t);

        // a kept old patch's data is in its new patch: its transfers (border points of the new
        //  patches touching it) read from there
        std::vector<std::size_t> kept_into(olds.size(), not_local);
        for (std::size_t d = 0; d < map.kept.size(); ++d)
            if (map.kept[d] != not_local)
                kept_into[map.kept[d]] = d;

        for (std::size_t s = 0; s < olds.size(); ++s)
        {
            DAMN_LOG_SCOPE(2, "damn::amr::regrid::old_patch"); // copied, packed, freed
            auto* src = olds[s] ? &*olds[s]
                                : (kept_into[s] != not_local ? &*news[kept_into[s]] : nullptr);
            if (!src)
                continue;
            for (auto const* t : from[s])
            {
                if (t->dst != not_local)
                    copy_transfer(map, *t, *src, make_new(t->dst));
                else
                    send(*t, *src, make_new);
            }
            olds[s].reset();
        }

        finish(make_new);

        LevelData<dim, PatchData> out{map.level, map.new_local, {}, old.prototype};
        set_prototype(out, make);
        out.data.reserve(news.size());
        for (std::size_t d = 0; d < news.size(); ++d)
        {
            auto& made = make_new(d); // made, if nothing reached it
            for_each_regridded<dim>(made, [](std::string_view, auto& res) {
                if constexpr (CellItemData<std::decay_t<decltype(res)>, dim>)
                    cell_items::appended(res); // items in from every transfer
            });
            out.data.emplace_back(std::move(made));
        }
        return out;
    }
} // namespace detail


// on one rank: throws if anything comes from, or must go to, another rank
template<std::size_t dim, typename PatchData, typename Make>
MKN_KUL_FN_SIG_RET LevelData<dim, PatchData> regrid(RegridMap<dim> const& map,
                                                    LevelData<dim, PatchData>&& old, Make&& make)
{
    for (auto const& t : map.transfers)
        if (t.src == not_local or t.dst == not_local)
            throw std::runtime_error("damn::amr::regrid: patches on other ranks, regrids between"
                                     " ranks are damn::mpi::regrid's");
    return detail::regrid(
        map, std::move(old), make, [](auto const&, auto&, auto&) {}, [](auto&) {});
}

template<std::size_t dim>
MKN_KUL_FN_SIG_RET bool same_decomposition(Level<dim> const& a, Level<dim> const& b)
{
    if (a.patches.size() != b.patches.size())
        return false;
    for (std::size_t i = 0; i < a.patches.size(); ++i)
        if (a.patches[i].box != b.patches[i].box or a.patches[i].rank != b.patches[i].rank)
            return false;
    return true;
}

} // namespace damn::amr

#endif // DAMN_AMR_REGRID_HPP
