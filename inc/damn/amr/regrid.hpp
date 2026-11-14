#ifndef DAMN_AMR_REGRID_HPP
#define DAMN_AMR_REGRID_HPP

// regridding a level: its data moved from one decomposition (boxes and owners) to another of the
// same cells, e.g. a rebalance (new owners), or the same cells chopped differently
//
//   auto map = make_regrid_map(h.levels[ln], new_level, rank);
//   ld       = regrid(map, std::move(ld), make_patch_data);   // damn::mpi::regrid over ranks
//   set_level(h, new_level);  n = make_neighbours(...);       // then refill ghosts
//
// a new patch with the same box and owner as an old one takes its data as is (moved, ghosts
// included). Otherwise new patches are made (make(patch)) and get their domain from the old
// patches overlapping or touching it, periodic images included (make_regrid_map's shifts):
//   BoxData:       the values on the domain points both share: domain_box(resource, box) of
//                  either, e.g. a primal node on the border of an old patch next to the new one,
//                  without shared cells
//   ParticleData:  the items whose cell is in the shared cells: domain items only, ghost items
//                  are outside every old domain
// for every resource of those kinds (for_each_resource); other resources are left as made.
// Ghosts are not filled: copy fill_ghosts (and the particle ghosts) on the new level after.
//
// memory: the old data is consumed. Old patches are visited in level order (along the curve the
// partition follows), each freed once its data is copied or packed; a new patch is made when its
// first data arrives. The peak is about the old level, not the old and the new together

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
#include "damn/amr/particle_data.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace damn::amr
{
// what of one old patch (or a periodic image of it) goes to one new patch
template<std::size_t dim>
struct Transfer
{
    Patch<dim> from{}, to{};
    core::Point<int, dim> shift{}; // the old patch's image is at from.box + shift
    std::optional<core::Box<int, dim>> cells{}; // (from.box + shift) * to.box, if they share any
    std::size_t src = not_local; // old LevelData index, if this rank holds from
    std::size_t dst = not_local; // new LevelData index, if this rank holds to
};

// what moves where in a regrid, from one rank's point of view. Both ranks of a pair list the
// transfers between them in the same order (old level order, then new level order): messages
// need no description of their content
template<std::size_t dim>
struct RegridMap
{
    int level = 0;
    int rank  = 0;
    std::vector<Patch<dim>> old_patches{}, new_patches{}; // the whole levels
    std::vector<Patch<dim>> old_local{}, new_local{};     // this rank's, in LevelData order
    std::vector<std::size_t> kept{}; // kept[new index]: the old index whose data it takes as is
    std::vector<Transfer<dim>> transfers{}; // from or to this rank, local or not
};


// shifts: the level's periodic images (periodic_shifts, the domain itself excluded)
template<std::size_t dim>
NO_DISCARD RegridMap<dim> make_regrid_map(Level<dim> const& old_level, Level<dim> const& new_level,
                                          int const rank,
                                          std::vector<core::Point<int, dim>> shifts = {})
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
                       {}};

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

    // overlapping or touching (a border's primal points): every rank lists the same, in the
    //  same order
    shifts.insert(shifts.begin(), core::Point<int, dim>{});
    for (auto const& o : old_level.patches)
        for (auto const& shift : shifts)
            for (std::size_t n = 0; n < new_level.patches.size(); ++n)
            {
                auto const& to = new_level.patches[n];
                if (is_kept[n] or (o.rank != rank and to.rank != rank))
                    continue;
                auto const image = core::shift(o.box, shift);
                if (core::grow(image, 1) * to.box)
                    map.transfers.push_back({o, to, shift, image * to.box,
                                             index_in(map.old_local, o.id),
                                             index_in(map.new_local, to.id)});
            }
    return map;
}


namespace detail
{
    // the domain points of a field resource a transfer copies
    template<std::size_t dim>
    NO_DISCARD std::optional<core::Box<int, dim>> shared_points(Transfer<dim> const& t,
                                                                auto const& res)
    {
        return domain_box(res, core::shift(t.from.box, t.shift)) * domain_box(res, t.to.box);
    }

    // fn(name, resource) for data's resources damn can regrid: BoxData and ParticleData
    template<std::size_t dim, typename PatchData>
    void for_each_regridded(PatchData& data, auto&& fn)
    {
        for_each_resource(data, [&](std::string_view const name, auto& resource) {
            using R = std::decay_t<decltype(resource)>;
            if constexpr (BoxData<R, dim> or ParticleData<R, dim>)
                fn(name, resource);
        });
    }

    // a transfer's data from its (local) source, appended to out, resource by resource:
    // field values in the box's point order, or the number of items then the items
    template<std::size_t dim, typename PatchData>
    void pack_transfer(Transfer<dim> const& t, PatchData& src, std::vector<std::byte>& out)
    {
        for_each_regridded<dim>(src, [&](std::string_view, auto const& res) {
            using R = std::decay_t<decltype(res)>;
            if constexpr (ParticleData<R, dim>)
            {
                auto const count_at = out.size();
                write(out, std::uint64_t{0});
                std::uint64_t count = 0;
                if (t.cells)
                    for_each_item(res, [&](auto const& item) {
                        if (core::isIn(cell(item) + t.shift, *t.cells))
                        {
                            write(out, item);
                            ++count;
                        }
                    });
                std::memcpy(out.data() + count_at, &count, sizeof(count));
            }
            else if (auto const box = shared_points(t, res))
            {
                using T = box_data_value_t<R const, dim>;
                out.reserve(out.size() + static_cast<std::size_t>(box->size()) * sizeof(T));
                core::for_each_point(*box, [&](auto const& p) {
                    write(out, T{value(res, t.from.box, p - t.shift)});
                });
            }
        });
    }

    // a transfer's data, packed by pack_transfer, into its (local) destination; at moves past it
    template<std::size_t dim, typename PatchData>
    void unpack_transfer(Transfer<dim> const& t, PatchData& dst, std::byte const*& at)
    {
        for_each_regridded<dim>(dst, [&](std::string_view, auto& res) {
            using R = std::decay_t<decltype(res)>;
            if constexpr (ParticleData<R, dim>)
            {
                using Item       = typename R::value_type;
                auto const count = read<std::uint64_t>(at);
                for (std::uint64_t i = 0; i < count; ++i)
                    append(res, read<Item>(at), t.shift);
            }
            else if (auto const box = shared_points(t, res))
            {
                using T = box_data_value_t<R, dim>;
                core::for_each_point(*box,
                                     [&](auto const& p) { value(res, t.to.box, p) = read<T>(at); });
            }
        });
    }

    // a transfer between two local patches, directly
    template<std::size_t dim, typename PatchData>
    void copy_transfer(Transfer<dim> const& t, PatchData& src, PatchData& dst)
    {
        for_each_regridded<dim>(dst, [&](std::string_view const name, auto& d) {
            using R          = std::decay_t<decltype(d)>;
            bool const found = with_resource(src, name, [&](auto const& s) {
                if constexpr (std::is_same_v<std::decay_t<decltype(s)>, R>)
                {
                    if constexpr (ParticleData<R, dim>)
                    {
                        if (t.cells)
                            for_each_item(s, [&](auto const& item) {
                                if (core::isIn(cell(item) + t.shift, *t.cells))
                                    append(d, item, t.shift);
                            });
                    }
                    else if (auto const box = shared_points(t, d))
                        core::for_each_point(*box, [&](auto const& p) {
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

    // the regrid of one rank's data: old patches visited in order, each freed once done.
    // send(transfer, old data, new_data) takes the transfers to other ranks' patches (packs
    // them); finish(new_data) receives the rest of what other ranks send. new_data(d) is new
    // local patch d's data, made on first use
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

        // transfers from each old local patch, in map order
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
                    copy_transfer(*t, *src, make_new(t->dst));
                else
                    send(*t, *src, make_new);
            }
            olds[s].reset();
        }

        finish(make_new);

        LevelData<dim, PatchData> out{map.level, map.new_local, {}};
        out.data.reserve(news.size());
        for (std::size_t d = 0; d < news.size(); ++d)
            out.data.emplace_back(std::move(make_new(d))); // made, if nothing reached it
        return out;
    }
} // namespace detail


// on one rank: throws if anything comes from, or must go to, another rank
template<std::size_t dim, typename PatchData, typename Make>
NO_DISCARD LevelData<dim, PatchData> regrid(RegridMap<dim> const& map,
                                            LevelData<dim, PatchData>&& old, Make&& make)
{
    for (auto const& t : map.transfers)
        if (t.src == not_local or t.dst == not_local)
            throw std::runtime_error("damn::amr::regrid: patches on other ranks, regrids between"
                                     " ranks are damn::mpi::regrid's");
    return detail::regrid(
        map, std::move(old), make, [](auto const&, auto&, auto&) {}, [](auto&) {});
}

// true if the levels are the same decomposition: nothing to regrid
template<std::size_t dim>
NO_DISCARD bool same_decomposition(Level<dim> const& a, Level<dim> const& b)
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
