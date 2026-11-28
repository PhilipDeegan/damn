#ifndef DAMN_AMR_DECOMPOSE_HPP
#define DAMN_AMR_DECOMPOSE_HPP

// patches and owners for a hierarchy's levels: doc/amr/decompose.md
// weigh may be collective: every rank calls it in the same order

#include "damn/core/def.hpp"
#include "damn/core/geometry/box.hpp"

#include "damn/amr/cost.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/keep_partition.hpp"
#include "damn/amr/partition.hpp"
#include "damn/amr/topology.hpp"

#include "mkn/kul/std/array.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace damn::amr
{
template<std::size_t dim>
struct Decompose
{
    Topology const& topology;
    Weigh<dim> weigh;
    std::array<int, dim> patch_size{}; // the largest, in the level's cells
    std::string partition = "levels";  // levels | keep | finest | aligned
    double spill          = -1.;       // keep, finest, aligned: spill's tol; < 0: no spill
    int block             = 8;         // cut_partition's block, cells per direction
    bool cut              = true;      // levels: cut patches on several ranks; false: whole
    std::vector<std::array<int, dim>> level_size{}; // per level, else patch_size
    std::vector<std::array<int, dim>> under_size{}; // per level, under the finer level; 0: none
    int near_band = 0;                              // finest in pieces: doc/amr/decompose.md
};

namespace detail
{
    // (size, under size), refined levels' in whole coarse cells
    template<std::size_t dim>
    std::pair<std::array<int, dim>, std::array<int, dim>> chop_sizes(Decompose<dim> const& in,
                                                                     int const ln, int const r)
    {
        auto const at = [&](auto const& v) {
            return v[std::min(static_cast<std::size_t>(ln), v.size() - 1)];
        };
        auto size  = in.level_size.empty() ? in.patch_size : at(in.level_size);
        auto under = in.under_size.empty() ? std::array<int, dim>{} : at(in.under_size);
        if (ln > 0)
            for (std::size_t d = 0; d < dim; ++d)
            {
                size[d] = std::max(r, size[d] / r * r);
                if (under[d] > 0)
                    under[d] = std::max(r, under[d] / r * r);
            }
        return {size, under};
    }

    // boxes split into (under, rest): under, their parts below the finer level's boxes
    // (coarsened by r), finer box after finer box; rest, the others
    template<std::size_t dim>
    std::pair<std::vector<core::Box<int, dim>>, std::vector<core::Box<int, dim>>>
    split_under(std::vector<core::Box<int, dim>> const& boxes,
                std::vector<core::Box<int, dim>> const& finer, int const r)
    {
        std::vector<core::Box<int, dim>> under, rest = boxes;
        for (auto const& f : finer)
        {
            auto const c = core::coarsen(f, r);
            std::vector<core::Box<int, dim>> left;
            for (auto const& b : rest)
            {
                if (auto const o = b * c)
                    under.push_back(*o);
                for (auto const& piece : core::minus_ordered(b, c))
                    left.push_back(piece);
            }
            rest = std::move(left);
        }
        return {under, rest};
    }

    // patches binned on a grid of cells, for the patches a box overlaps
    template<std::size_t dim>
    class PatchIndex
    {
    public:
        PatchIndex(std::vector<core::Box<int, dim>> const& boxes)
            : boxes_{boxes}
        {
            if (boxes.empty())
                return;
            lower_                     = boxes[0].lower;
            std::array<int, dim> upper = boxes[0].upper;
            for (auto const& b : boxes)
                for (std::size_t d = 0; d < dim; ++d)
                {
                    lower_[d] = std::min(lower_[d], b.lower[d]);
                    upper[d]  = std::max(upper[d], b.upper[d]);
                    bin_[d]   = std::max(bin_[d], b.shape()[d]);
                }
            for (std::size_t d = 0; d < dim; ++d)
                n_[d] = (upper[d] - lower_[d]) / bin_[d] + 1;
            std::size_t bins = 1;
            for (auto const n : n_)
                bins *= static_cast<std::size_t>(n);
            cells_.resize(bins);
            for (std::size_t i = 0; i < boxes.size(); ++i)
                visit(boxes[i], [&](std::size_t const bin) { cells_[bin].push_back(i); });
            seen_.assign(boxes.size(), 0);
        }

        // f(i, cells) for each patch i overlapping b, cells: their overlap's
        template<typename F>
        void overlaps(core::Box<int, dim> const& b, F&& f)
        {
            if (boxes_.empty())
                return;
            ++stamp_;
            visit(b, [&](std::size_t const bin) {
                for (auto const i : cells_[bin])
                {
                    if (seen_[i] == stamp_)
                        continue;
                    seen_[i] = stamp_;
                    if (auto const o = b * boxes_[i])
                        f(i, static_cast<double>(o->size()));
                }
            });
        }

    private:
        template<typename F>
        void visit(core::Box<int, dim> const& b, F&& f) const
        {
            std::array<int, dim> first{}, last{};
            for (std::size_t d = 0; d < dim; ++d)
            {
                first[d] = std::clamp((b.lower[d] - lower_[d]) / bin_[d], 0, n_[d] - 1);
                last[d]  = std::clamp((b.upper[d] - lower_[d]) / bin_[d], 0, n_[d] - 1);
                if (b.upper[d] < lower_[d])
                    return;
            }
            auto k = first;
            while (true)
            {
                std::size_t flat = 0, stride = 1;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    flat += static_cast<std::size_t>(k[d]) * stride;
                    stride *= static_cast<std::size_t>(n_[d]);
                }
                f(flat);
                std::size_t d = 0;
                for (; d < dim; ++d)
                {
                    if (++k[d] <= last[d])
                        break;
                    k[d] = first[d];
                }
                if (d == dim)
                    break;
            }
        }

        std::vector<core::Box<int, dim>> const& boxes_;
        std::array<int, dim> lower_{};
        std::array<int, dim> bin_ = mkn::kul::ConstArray<int, dim>(1);
        std::array<int, dim> n_{};
        std::vector<std::vector<std::size_t>> cells_{};
        std::vector<std::size_t> seen_{};
        std::size_t stamp_ = 0;
    };

    // per box (in the index's cells) the rank holding most of its cells among the indexed
    // patches' ranks (ties: the lower rank), `none` where it overlaps none
    template<std::size_t dim>
    std::vector<int> majority(std::vector<core::Box<int, dim>> const& boxes,
                              std::vector<core::Box<int, dim>> const& patches,
                              std::vector<int> const& ranks, int const n_ranks, int const none)
    {
        PatchIndex<dim> index{patches};
        std::vector<double> cells(static_cast<std::size_t>(n_ranks));
        std::vector<int> out;
        out.reserve(boxes.size());
        for (auto const& b : boxes)
        {
            std::fill(cells.begin(), cells.end(), 0.);
            bool any = false;
            index.overlaps(b, [&](std::size_t const i, double const c) {
                cells[static_cast<std::size_t>(ranks[i])] += c;
                any = true;
            });
            if (!any)
            {
                out.push_back(none);
                continue;
            }
            out.push_back(static_cast<int>(
                std::distance(cells.begin(), std::max_element(cells.begin(), cells.end()))));
        }
        return out;
    }

    template<std::size_t dim>
    void check_whole_coarse(int const ln, std::vector<core::Box<int, dim>> const& patches,
                            int const r)
    {
        if (ln == 0)
            return;
        for (auto const& p : patches)
            for (std::size_t d = 0; d < dim; ++d)
                if (p.lower[d] % r != 0 or p.shape()[d] % r != 0)
                    throw std::runtime_error("damn::amr::decompose: a level " + std::to_string(ln)
                                             + " patch not in whole coarse cells");
    }

    // a level's patches: its boxes chopped; with an under size, the parts under the finer
    // level's boxes first, chopped at it, then the rest
    template<std::size_t dim>
    std::vector<core::Box<int, dim>> chop_level(Decompose<dim> const& in, int const ln, int const r,
                                                std::vector<core::Box<int, dim>> const& boxes,
                                                std::vector<core::Box<int, dim>> const* finer)
    {
        auto const [size, under] = chop_sizes(in, ln, r);
        auto const chop          = [&](std::vector<core::Box<int, dim>> const& bs, auto const& s) {
            std::vector<core::Box<int, dim>> patches;
            for (auto const& box : bs)
                for (auto const& b : chop_evenly(box, s, ln > 0 ? r : 1))
                    patches.push_back(b);
            return patches;
        };
        if (!finer or under[0] <= 0)
            return chop(boxes, size);
        auto const [below, rest] = split_under(boxes, *finer, r);
        auto patches             = chop(below, under);
        for (auto const& p : chop(rest, size))
            patches.push_back(p);
        return patches;
    }
} // namespace detail

namespace detail
{
    // "finest" and "aligned": every level from boxes known
    template<std::size_t dim, typename Lattice>
    std::vector<Level<dim>>
    decompose_together(Hierarchy<dim, Lattice> const& h, int const from,
                       std::vector<std::vector<core::Box<int, dim>>> const& boxes,
                       Decompose<dim> const& in)
    {
        using Box_t        = core::Box<int, dim>;
        auto const n       = boxes.size();
        auto const n_ranks = static_cast<int>(in.topology.ranks.size());
        auto const r       = static_cast<int>(h.ratio);
        auto const cap     = node_share_capacity(in.topology);

        std::vector<std::vector<Box_t>> patches(n);
        std::vector<std::vector<double>> weights(n);
        for (std::size_t k = 0; k < n; ++k)
        {
            auto const ln = from + static_cast<int>(k);
            patches[k]    = chop_level(in, ln, r, boxes[k], k + 1 < n ? &boxes[k + 1] : nullptr);
            check_whole_coarse(ln, patches[k], r);
            weights[k] = in.weigh(ln, patches[k]);
        }
        auto const subset = [](auto const& v, std::vector<std::size_t> const& idx) {
            std::remove_cvref_t<decltype(v)> out;
            for (auto const i : idx)
                out.push_back(v[i]);
            return out;
        };
        // a level's patches to the owners `ranks` gives, those at `none` along the curve
        auto const fill_rest = [&](std::size_t const k, std::vector<int>& ranks) {
            std::vector<std::size_t> rest;
            for (std::size_t i = 0; i < ranks.size(); ++i)
                if (ranks[i] < 0)
                    rest.push_back(i);
            if (rest.empty())
                return;
            auto const curve = partition(subset(patches[k], rest), subset(weights[k], rest), cap);
            for (std::size_t j = 0; j < rest.size(); ++j)
                ranks[rest[j]] = curve[j];
        };
        auto const spilled = [&](std::size_t const k, std::vector<int> ranks) {
            return in.spill >= 0
                       ? amr::spill(patches[k], std::move(ranks), weights[k], in.topology, in.spill)
                       : ranks;
        };
        // each patch's box in the finer (refined) or coarser (coarsened) level's cells
        auto const refined = [&](std::vector<Box_t> const& bs) {
            std::vector<Box_t> out;
            for (auto const& b : bs)
                out.push_back(core::refine(b, r));
            return out;
        };
        auto const coarsened = [&](std::vector<Box_t> const& bs) {
            std::vector<Box_t> out;
            for (auto const& b : bs)
                out.push_back(core::coarsen(b, r));
            return out;
        };
        auto const fixed_coarser = [&]() -> std::pair<std::vector<Box_t>, std::vector<int>> {
            std::pair<std::vector<Box_t>, std::vector<int>> out;
            if (from == 0 or static_cast<std::size_t>(from - 1) >= h.levels.size())
                return out;
            for (auto const& p : h.levels[static_cast<std::size_t>(from - 1)].patches)
            {
                out.first.push_back(p.box);
                out.second.push_back(p.rank);
            }
            return out;
        };

        std::vector<std::vector<int>> ranks(n);
        if (in.partition == "finest")
        {
            auto const [below, below_ranks] = fixed_coarser();
            auto const under_fixed          = [&](std::vector<int>& rk) {
                if (below.empty())
                    return;
                std::vector<std::size_t> rest;
                for (std::size_t i = 0; i < rk.size(); ++i)
                    if (rk[i] < 0)
                        rest.push_back(i);
                auto const owners = majority(coarsened(subset(patches[0], rest)), below,
                                             below_ranks, n_ranks, -1);
                for (std::size_t j = 0; j < rest.size(); ++j)
                    rk[rest[j]] = owners[j];
            };
            for (std::size_t k = n; k-- > 0;)
            {
                if (k + 1 == n and (k > 0 or below.empty()))
                {
                    ranks[k] = partition(patches[k], weights[k], cap);
                    continue;
                }
                ranks[k] = k + 1 < n ? majority(refined(patches[k]), patches[k + 1], ranks[k + 1],
                                                n_ranks, -1)
                                     : std::vector<int>(patches[k].size(), -1);
                if (k == 0)
                    under_fixed(ranks[k]);
                fill_rest(k, ranks[k]);
                ranks[k] = spilled(k, std::move(ranks[k]));
            }
        }
        else // aligned
        {
            auto const [below, below_ranks] = fixed_coarser();
            for (std::size_t k = 0; k < n; ++k)
            {
                if (k == 0 and below.empty())
                {
                    ranks[k] = partition(patches[k], weights[k], cap);
                    continue;
                }
                auto const& cb = k == 0 ? below : patches[k - 1];
                auto const& cr = k == 0 ? below_ranks : ranks[k - 1];
                ranks[k]       = majority(coarsened(patches[k]), cb, cr, n_ranks, 0);
                ranks[k]       = spilled(k, std::move(ranks[k]));
            }
        }
        std::vector<Level<dim>> out;
        for (std::size_t k = 0; k < n; ++k)
            out.push_back(make_level(from + static_cast<int>(k), patches[k], ranks[k]));
        return out;
    }
} // namespace detail

namespace detail
{
    // a coarse patch is never given whole to one rank for the fine cells above part of it
    template<std::size_t dim, typename Lattice>
    std::vector<Level<dim>>
    finest_in_pieces(Hierarchy<dim, Lattice> const& h, int const from,
                     std::vector<std::vector<core::Box<int, dim>>> const& boxes,
                     Decompose<dim> const& in)
    {
        using Box_t             = core::Box<int, dim>;
        auto const n            = boxes.size();
        auto const n_ranks      = static_cast<int>(in.topology.ranks.size());
        auto const r            = static_cast<int>(h.ratio);
        auto const cap          = node_share_capacity(in.topology);
        auto const* const below = from > 0 and static_cast<std::size_t>(from - 1) < h.levels.size()
                                      ? &h.levels[static_cast<std::size_t>(from - 1)].patches
                                      : nullptr;

        std::vector<Level<dim>> out(n);
        for (std::size_t k = n; k-- > 0;)
        {
            auto const ln   = from + static_cast<int>(k);
            auto const chop = [&](std::vector<Box_t> const& bs) {
                return chop_level<dim>(in, ln, r, bs, nullptr);
            };
            auto const b     = ln > 0 ? std::max(r, in.block / r * r) : in.block;
            auto const block = mkn::kul::ConstArray<int, dim>(b);
            auto const fresh = [&](std::vector<Box_t> const& rest) {
                auto const patches = chop(rest);
                auto const cut
                    = cut_partition(patches, block, in.weigh(ln, blocks(patches, block)), cap);
                std::vector<OwnedBox<dim>> owned;
                for (std::size_t i = 0; i < cut.boxes.size(); ++i)
                    owned.push_back({cut.boxes[i], cut.ranks[i]});
                return owned;
            };
            // rounded out to whole coarser cells on a refined level, so every piece is; made
            //  disjoint because keep takes disjoint owners (a cell goes to the first)
            std::vector<Patch<dim>> above;
            auto const claim = [&](Box_t const& box, Patch<dim> const& p) {
                std::vector<Box_t> parts{box};
                for (auto const& a : above)
                {
                    std::vector<Box_t> left;
                    for (auto const& part : parts)
                        for (auto const& piece : core::minus_ordered(part, a.box))
                            left.push_back(piece);
                    parts = std::move(left);
                }
                for (auto const& part : parts)
                    above.push_back({part, p.rank, p.id});
            };
            auto const in_level = [&](Box_t box) {
                if (ln > 0)
                    box = core::refine(core::coarsen(box, r), r);
                return box;
            };
            if (k + 1 < n)
            {
                for (auto const& p : out[k + 1].patches)
                    claim(in_level(core::coarsen(p.box, r)), p);
                if (in.near_band > 0 and ln == 0) // refined levels: nested, mostly near
                    for (auto const& p : out[k + 1].patches)
                        claim(in_level(core::grow(core::coarsen(p.box, r), in.near_band)), p);
            }
            std::vector<Patch<dim>> const fixed
                = k == 0 and below ? *below : std::vector<Patch<dim>>{};
            auto const pieces = keep(boxes[k], above, fixed, h.ratio, fresh);

            // spill on blocks: the closest rank with room takes one, so near pieces stay near
            //  their owner
            auto per_rank = owned_boxes(pieces, n_ranks);
            if (in.spill >= 0)
            {
                std::vector<Box_t> bs;
                std::vector<int> owners;
                for (std::size_t q = 0; q < per_rank.size(); ++q)
                    for (auto const& box : per_rank[q])
                        for (auto const& piece : blocks_of(box, block))
                        {
                            bs.push_back(piece);
                            owners.push_back(static_cast<int>(q));
                        }
                owners = amr::spill(bs, owners, in.weigh(ln, bs), in.topology, in.spill);
                std::vector<OwnedBox<dim>> spilled;
                for (std::size_t i = 0; i < bs.size(); ++i)
                    spilled.push_back({bs[i], owners[i]});
                per_rank = owned_boxes(spilled, n_ranks);
            }
            std::vector<Box_t> patches;
            std::vector<int> ranks;
            for (std::size_t q = 0; q < per_rank.size(); ++q)
                for (auto const& p : chop(per_rank[q]))
                {
                    patches.push_back(p);
                    ranks.push_back(static_cast<int>(q));
                }
            check_whole_coarse(ln, patches, r);
            out[k] = make_level(ln, patches, ranks);
        }
        return out;
    }
} // namespace detail

template<std::size_t dim, typename Lattice>
MKN_KUL_FN_SIG_RET Level<dim>
decompose_level(Hierarchy<dim, Lattice> const& h, int const ln,
                std::vector<core::Box<int, dim>> const& boxes, Decompose<dim> const& in,
                std::vector<core::Box<int, dim>> const* finer = nullptr)
{
    using Box_t = core::Box<int, dim>;
    if (in.partition == "finest" or in.partition == "aligned")
    {
        if (ln > 0 and static_cast<std::size_t>(ln - 1) < h.levels.size())
            return in.partition == "finest" and in.cut and in.topology.ranks.size() > 1
                       ? detail::finest_in_pieces(h, ln, {boxes}, in).front()
                       : detail::decompose_together(h, ln, {boxes}, in).front();
        auto as_levels      = in;
        as_levels.partition = "levels";
        return decompose_level(h, ln, boxes, as_levels, finer);
    }
    if (in.partition != "levels" and in.partition != "keep")
        throw std::runtime_error("damn::amr::decompose: partition levels, keep, finest or "
                                 "aligned, not "
                                 + in.partition);
    auto const n_ranks = static_cast<int>(in.topology.ranks.size());
    auto const r       = static_cast<int>(h.ratio);

    // evenly: no patch left with a few cells
    auto const chop = [&](std::vector<Box_t> const& bs) {
        return detail::chop_level<dim>(in, ln, r, bs, finer);
    };
    auto const capacity = node_share_capacity(in.topology);
    auto const curve    = [&](std::vector<Box_t> const& patches) {
        return partition(patches, in.weigh(ln, patches), capacity);
    };
    auto const whole_coarse
        = [&](std::vector<Box_t> const& patches) { detail::check_whole_coarse(ln, patches, r); };

    auto const l          = static_cast<std::size_t>(ln);
    auto const* const old = l < h.levels.size() ? &h.levels[l].patches : nullptr;
    auto const* const coarser
        = ln > 0 and l - 1 < h.levels.size() ? &h.levels[l - 1].patches : nullptr;
    if (in.partition != "keep" or (!old and !coarser))
    {
        auto const patches = chop(boxes);
        whole_coarse(patches);
        if (n_ranks == 1 or !in.cut)
            return make_level(ln, patches, curve(patches));
        // cut so every rank gets work even when the largest patch is the domain; refined
        //  levels' blocks are whole coarser cells from a patch's lower corner (itself one)
        auto const b     = ln > 0 ? std::max(r, in.block / r * r) : in.block;
        auto const block = mkn::kul::ConstArray<int, dim>(b);
        auto const cut
            = cut_partition(patches, block, in.weigh(ln, blocks(patches, block)), capacity);
        whole_coarse(cut.boxes);
        return make_level(ln, cut.boxes, cut.ranks);
    }

    using Patches     = std::vector<Patch<dim>>;
    auto const pieces = keep(boxes, old ? *old : Patches{}, coarser ? *coarser : Patches{}, h.ratio,
                             [&](std::vector<Box_t> const& rest) {
                                 auto const patches = chop(rest);
                                 auto const ranks   = curve(patches);
                                 std::vector<OwnedBox<dim>> out;
                                 for (std::size_t i = 0; i < patches.size(); ++i)
                                     out.push_back({patches[i], ranks[i]});
                                 return out;
                             });
    std::vector<Box_t> patches;
    std::vector<int> ranks;
    auto const per_rank = owned_boxes(pieces, n_ranks);
    for (std::size_t k = 0; k < per_rank.size(); ++k)
        for (auto const& b : chop(per_rank[k]))
        {
            patches.push_back(b);
            ranks.push_back(static_cast<int>(k));
        }
    if (in.spill >= 0)
        ranks = amr::spill(patches, ranks, in.weigh(ln, patches), in.topology, in.spill);
    whole_coarse(patches);
    return make_level(ln, patches, ranks);
}



template<std::size_t dim, typename Lattice>
MKN_KUL_FN_SIG_RET std::vector<Level<dim>>
decompose(Hierarchy<dim, Lattice> h, int const from,
          std::vector<std::vector<core::Box<int, dim>>> const& boxes, Decompose<dim> const& in)
{
    if (static_cast<std::size_t>(from) > h.levels.size())
        throw std::runtime_error("damn::amr::decompose: level " + std::to_string(from)
                                 + " without its coarser level");
    if (in.partition == "finest" and in.cut and in.topology.ranks.size() > 1)
        return detail::finest_in_pieces(h, from, boxes, in);
    if (in.partition == "finest" or in.partition == "aligned")
        return detail::decompose_together(h, from, boxes, in);
    std::vector<Level<dim>> out;
    for (std::size_t k = 0; k < boxes.size(); ++k)
    {
        auto const ln = from + static_cast<int>(k);
        auto const l  = static_cast<std::size_t>(ln);
        out.push_back(
            decompose_level(h, ln, boxes[k], in, k + 1 < boxes.size() ? &boxes[k + 1] : nullptr));
        // the next level's coarser one, in place: the finer old levels stay for "keep"
        if (l < h.levels.size())
            h.levels[l] = out.back();
        else
            h.levels.push_back(out.back());
    }
    return out;
}

} // namespace damn::amr

#endif // DAMN_AMR_DECOMPOSE_HPP
