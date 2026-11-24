#ifndef DAMN_AMR_HIERARCHY_HPP
#define DAMN_AMR_HIERARCHY_HPP

// a patch hierarchy as data, and free functions over it
//
//   Hierarchy: level 0 domain, mesh size, origin, periodicity, refinement ratio, levels; its
//              Lattice (damn/amr/lattice.hpp) a tag: Cartesian by default, or Kelvin
//   Level:     every patch of the level, whichever rank owns it (same on every rank)
//   Patch:     a cell box in its level's index space, owner rank, id unique on the level
//
// level ln's index space is level 0's refined by ratio^ln; boxes are cell boxes
// nothing here communicates: ranks are only numbers, see damn/mpi for that

#include "damn/core/def.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/amr/lattice.hpp"
#include "damn/amr/partition.hpp"
#include "mkn/kul/std/array.hpp"

#include <array>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <initializer_list>

namespace damn::amr
{
template<std::size_t dim>
struct Patch
{
    core::Box<int, dim> box{}; // cells, level index space
    int rank         = 0;      // owner
    std::uint32_t id = 0;      // unique on the level

    bool operator==(Patch const&) const = default;
};

template<std::size_t dim>
struct Level
{
    int number = 0;
    std::vector<Patch<dim>> patches{}; // all of them, local or not
};

template<std::size_t dim, typename Lattice_ = Cartesian>
struct Hierarchy
{
    auto static constexpr dimension = dim;
    using Lattice                   = Lattice_;

    core::Box<int, dim> domain{};      // level 0 cells
    std::array<double, dim> dx{};      // level 0 mesh size
    core::Point<double, dim> origin{}; // physical position of domain.lower's lower corner
    std::array<bool, dim> periodic{};  // per direction
    int max_levels = 1;                // levels allowed, including 0
    int ratio      = 2;                // between consecutive levels
    std::vector<Level<dim>> levels{};  // 0 .. finest
};


// ---------------------------------------------------------------- levels

template<std::size_t dim, typename Lattice>
NO_DISCARD int n_levels(Hierarchy<dim, Lattice> const& h)
{
    return static_cast<int>(h.levels.size());
}

template<std::size_t dim, typename Lattice>
NO_DISCARD int finest_level(Hierarchy<dim, Lattice> const& h)
{
    return n_levels(h) - 1;
}

template<std::size_t dim, typename Lattice>
NO_DISCARD bool level_exists(Hierarchy<dim, Lattice> const& h, int const ln)
{
    return ln >= 0 and ln < n_levels(h);
}

// level ln's index space is ratio_to_level_0 times finer than level 0's
template<std::size_t dim, typename Lattice>
NO_DISCARD int ratio_to_level_0(Hierarchy<dim, Lattice> const& h, int const ln)
{
    int r = 1;
    for (int i = 0; i < ln; ++i)
        r *= h.ratio;
    return r;
}

// the domain in level ln's index space
template<std::size_t dim, typename Lattice>
NO_DISCARD core::Box<int, dim> level_domain(Hierarchy<dim, Lattice> const& h, int const ln)
{
    return core::refine(h.domain, ratio_to_level_0(h, ln));
}

template<std::size_t dim, typename Lattice>
NO_DISCARD std::array<double, dim> mesh_size(Hierarchy<dim, Lattice> const& h, int const ln)
{
    auto dx      = h.dx;
    auto const r = ratio_to_level_0(h, ln);
    for (auto& d : dx)
        d /= r;
    return dx;
}

// physical position of the lower corner of cell `cell` of level ln
template<std::size_t dim, typename Lattice>
NO_DISCARD core::Point<double, dim> position(Hierarchy<dim, Lattice> const& h, int const ln,
                                             core::Point<int, dim> const& cell)
{
    auto const dx    = mesh_size(h, ln);
    auto const lower = level_domain(h, ln).lower;
    core::Point<double, dim> x;
    for (std::size_t i = 0; i < dim; ++i)
        x[i] = h.origin[i] + (cell[i] - lower[i]) * dx[i];
    return x;
}

// physical position of a Kelvin site of cell `cell` of level ln: the cell's lower corner, or its
//  centre
template<std::size_t dim, typename Lattice>
    requires KelvinLattice<Lattice>
NO_DISCARD core::Point<double, dim> position(Hierarchy<dim, Lattice> const& h, int const ln,
                                             core::Point<int, dim> const& cell, Site const site)
{
    auto x = position(h, ln, cell);
    if (site == Site::body)
    {
        auto const dx = mesh_size(h, ln);
        for (std::size_t i = 0; i < dim; ++i)
            x[i] += dx[i] / 2;
    }
    return x;
}


// throws if the refinement ratio does not suit the lattice: Kelvin's must be odd
template<std::size_t dim, typename Lattice>
    requires KelvinLattice<Lattice>
void check_ratio(Hierarchy<dim, Lattice> const& h)
{
    if (h.ratio < 1 or h.ratio % 2 == 0)
        throw std::runtime_error("damn::amr: a Kelvin lattice's refinement ratio is odd, not "
                                 + std::to_string(h.ratio));
}

template<std::size_t dim, typename Lattice>
void check_ratio(Hierarchy<dim, Lattice> const& h)
{
    if (h.ratio < 1)
        throw std::runtime_error("damn::amr: refinement ratio " + std::to_string(h.ratio));
}

// throws if level's patches are not inside the domain, not of whole cells of the level below,
//  overlap each other, or ids repeat, or the ratio does not suit the lattice
template<std::size_t dim, typename Lattice>
void check_level(Hierarchy<dim, Lattice> const& h, Level<dim> const& level)
{
    auto const fail = [&](std::string const& why) {
        throw std::runtime_error("damn::amr level " + std::to_string(level.number) + ": " + why);
    };

    if (level.number < 0 or level.number >= h.max_levels)
        fail("level number out of range");
    if (level.number > n_levels(h))
        fail("coarser level missing");
    if (level.patches.empty())
        fail("no patches");
    check_ratio(h);

    auto const domain = level_domain(h, level.number);
    for (std::size_t i = 0; i < level.patches.size(); ++i)
    {
        auto const& p = level.patches[i];
        if (auto const in = p.box * domain; !in or *in != p.box)
            fail("patch outside the domain");
        if (level.number > 0 and core::refine(core::coarsen(p.box, h.ratio), h.ratio) != p.box)
            fail("patch not of whole coarser cells");
        for (std::size_t j = i + 1; j < level.patches.size(); ++j)
        {
            if (p.box * level.patches[j].box)
                fail("patches overlap");
            if (p.id == level.patches[j].id)
                fail("patch ids repeat");
        }
    }
}

// sets level.number, replacing it if it exists; finer levels are removed, they were built
// on the replaced one
template<std::size_t dim, typename Lattice>
void set_level(Hierarchy<dim, Lattice>& h, Level<dim> level)
{
    check_level(h, level);
    auto const ln = level.number;
    h.levels.resize(ln); // drops ln and finer
    h.levels.emplace_back(std::move(level));
}

template<std::size_t dim, typename Lattice>
void remove_finer_levels(Hierarchy<dim, Lattice>& h, int const ln)
{
    if (ln + 1 < n_levels(h))
        h.levels.resize(ln + 1);
}

// a level of the given boxes, ids 0.., owned per `ranks` (all rank 0 if empty)
template<std::size_t dim>
NO_DISCARD Level<dim> make_level(int const ln, std::vector<core::Box<int, dim>> const& boxes,
                                 std::vector<int> const& ranks = {})
{
    if (!ranks.empty() and ranks.size() != boxes.size())
        throw std::runtime_error("damn::amr::make_level: one rank per box");

    Level<dim> level{ln, {}};
    level.patches.reserve(boxes.size());
    for (std::size_t i = 0; i < boxes.size(); ++i)
        level.patches.push_back(
            {boxes[i], ranks.empty() ? 0 : ranks[i], static_cast<std::uint32_t>(i)});
    return level;
}


// ---------------------------------------------------------------- patches

// fn(patch) for the patches of level owned by rank
template<std::size_t dim>
void for_each_local_patch(Level<dim> const& level, int const rank, auto&& fn)
{
    for (auto const& patch : level.patches)
        if (patch.rank == rank)
            fn(patch);
}

template<std::size_t dim>
NO_DISCARD std::vector<Patch<dim>> local_patches(Level<dim> const& level, int const rank)
{
    std::vector<Patch<dim>> local;
    for_each_local_patch(level, rank, [&](auto const& p) { local.push_back(p); });
    return local;
}

template<std::size_t dim>
NO_DISCARD Patch<dim> const& patch_with_id(Level<dim> const& level, std::uint32_t const id)
{
    for (auto const& patch : level.patches)
        if (patch.id == id)
            return patch;
    throw std::runtime_error("damn::amr: no patch with id " + std::to_string(id) + " on level "
                             + std::to_string(level.number));
}


// ---------------------------------------------------------------- decomposition

// box cut into boxes of at most `size` cells per direction, in lexicographic order (x fastest)
template<std::size_t dim>
NO_DISCARD std::vector<core::Box<int, dim>> chop(core::Box<int, dim> const& box,
                                                 std::array<int, dim> const& size)
{
    std::array<int, dim> n{};
    for (std::size_t i = 0; i < dim; ++i)
    {
        if (size[i] < 1)
            throw std::runtime_error("damn::amr::chop: size < 1");
        auto const width = box.upper[i] - box.lower[i] + 1;
        n[i]             = (width + size[i] - 1) / size[i];
    }

    std::size_t total = 1;
    for (auto const ni : n)
        total *= ni;

    std::vector<core::Box<int, dim>> boxes;
    boxes.reserve(total);
    for (std::size_t flat = 0; flat < total; ++flat)
    {
        auto rest = flat;
        core::Box<int, dim> b{box};
        for (std::size_t i = 0; i < dim; ++i)
        {
            auto const k = static_cast<int>(rest % n[i]);
            rest /= n[i];
            b.lower[i] = box.lower[i] + k * size[i];
            b.upper[i] = std::min(b.lower[i] + size[i] - 1, box.upper[i]);
        }
        boxes.push_back(b);
    }
    return boxes;
}

// how far past chop_evenly's size a box may go: size is a hint, not a bound
double constexpr chop_size_slack = 1.25;

// box cut per direction into the count of boxes nearest width / size, more only if a box would be
// wider than chop_size_slack * size, the cells shared evenly between them in units of
// `granularity` cells from the box's lower corner (e.g. the refinement ratio: refined boxes of
// whole coarser cells; size and its slack are rounded to whole units), the leftover units one
// each to the first boxes (SAMRAI's chop on one rank), so no box is left with a few cells.
// Size 20: 100 cells: 20 x 5; 45: 23, 22 (chop: 20, 20, 5); 24: 24; 26: 13, 13. Size 60,
// granularity 2: 122 cells: 62, 60 (chop: 60, 60, 2)
template<std::size_t dim>
NO_DISCARD std::vector<core::Box<int, dim>> chop_evenly(core::Box<int, dim> const& box,
                                                        std::array<int, dim> const& size,
                                                        int const granularity = 1)
{
    if (granularity < 1)
        throw std::runtime_error("damn::amr::chop_evenly: granularity < 1");
    std::array<std::vector<int>, dim> cuts; // per direction: each box's lower, then upper + 1
    for (std::size_t i = 0; i < dim; ++i)
    {
        if (size[i] < 1)
            throw std::runtime_error("damn::amr::chop_evenly: size < 1");
        auto const width = box.upper[i] - box.lower[i] + 1;
        auto const units = (width + granularity - 1) / granularity;
        auto const hint  = std::max(1, (2 * size[i] + granularity) / (2 * granularity)); // round
        auto const most = std::max(hint, static_cast<int>(chop_size_slack * size[i] / granularity));
        auto const nearest = (2 * units + hint) / (2 * hint); // round(units / hint)
        auto const n       = std::max({1, nearest, (units + most - 1) / most});
        for (int k = 0; k < n; ++k)
            cuts[i].push_back(box.lower[i] // ceil(k units / n)
                              + granularity
                                    * static_cast<int>((static_cast<long>(k) * units + n - 1) / n));
        cuts[i].push_back(box.upper[i] + 1);
    }

    std::size_t total = 1;
    for (auto const& c : cuts)
        total *= c.size() - 1;
    std::vector<core::Box<int, dim>> boxes;
    boxes.reserve(total);
    for (std::size_t flat = 0; flat < total; ++flat)
    {
        auto rest = flat;
        core::Box<int, dim> b{box};
        for (std::size_t i = 0; i < dim; ++i)
        {
            auto const n = cuts[i].size() - 1;
            auto const k = rest % n;
            rest /= n;
            b.lower[i] = cuts[i][k];
            b.upper[i] = cuts[i][k + 1] - 1;
        }
        boxes.push_back(b);
    }
    return boxes;
}

// level 0 as the domain chopped into patches of `size`, partitioned over n_ranks by cells along
// the Hilbert curve (partition.hpp; for other weights: make_level(0, boxes, partition(...)))
template<std::size_t dim, typename Lattice>
NO_DISCARD Level<dim> uniform_level_0(Hierarchy<dim, Lattice> const& h,
                                      std::array<int, dim> const& size, int const n_ranks = 1)
{
    auto const boxes = chop(h.domain, size);
    return make_level(0, boxes, partition(boxes, cell_weights(boxes), n_ranks));
}


// ---------------------------------------------------------------- periodicity

// offsets, in level ln's index space, of the periodic images of the domain next to it
// (not the domain itself): 3^p - 1 of them for p periodic directions
template<std::size_t dim, typename Lattice>
NO_DISCARD std::vector<core::Point<int, dim>> periodic_shifts(Hierarchy<dim, Lattice> const& h,
                                                              int const ln)
{
    auto const shape = level_domain(h, ln).shape();

    std::vector<core::Point<int, dim>> shifts{core::Point<int, dim>{}};
    for (std::size_t i = 0; i < dim; ++i)
    {
        if (!h.periodic[i])
            continue;
        auto const n = shifts.size();
        for (std::size_t s = 0; s < n; ++s)
            for (int const sign : {-1, 1})
            {
                auto shift = shifts[s];
                shift[i]   = sign * shape[i];
                shifts.push_back(shift);
            }
    }
    shifts.erase(shifts.begin()); // the domain itself
    return shifts;
}

// the parts of box (level ln index space) that level ln's patches cover, periodic images
// included, e.g. the ghost cells of a patch that are not level border
template<std::size_t dim, typename Lattice>
NO_DISCARD std::vector<core::Box<int, dim>>
covered_by_level(Hierarchy<dim, Lattice> const& h, int const ln, core::Box<int, dim> const& box)
{
    std::vector<core::Point<int, dim>> shifts{core::Point<int, dim>{}};
    for (auto const& s : periodic_shifts(h, ln))
        shifts.push_back(s);

    std::vector<core::Box<int, dim>> covered;
    for (auto const& patch : h.levels.at(ln).patches)
        for (auto const& shift : shifts)
            if (auto const overlap = box * core::shift(patch.box, shift))
                covered.push_back(*overlap);
    return covered;
}

// the cells of box grown by width (a patch's ghost box, level ln index space) another patch of
//  level ln owns, periodic images included. A cell's owner is, of the patch images holding it
//  in their box grown by width, the nearest to it (the L1 distance from the cell to the image's
//  box), the first in box order (core::lexicographic_less) on a tie: every holder agrees (both
//  are translation invariant, the same on every rank). A level ghost particle held by its
//  cell's owner only (PHAREHUB/PHARE#1580's owner, nearest rather than first): it stays in its
//  holder's ghost box the longest, e.g. a cell diagonal to one patch's corner and next to
//  another's side is the latter's. As boxes of cells, along the last direction
template<std::size_t dim, typename Lattice>
NO_DISCARD std::vector<core::Box<int, dim>>
owned_elsewhere(Hierarchy<dim, Lattice> const& h, int const ln, core::Box<int, dim> const& box,
                int const width)
{
    using Box_t   = core::Box<int, dim>;
    using Point_t = core::Point<int, dim>;
    auto const w  = mkn::kul::ConstArray<int, dim>(width);

    std::vector<Point_t> shifts{Point_t{}};
    for (auto const& s : periodic_shifts(h, ln))
        shifts.push_back(s);

    auto const grown = core::grow(box, w);
    std::vector<Box_t> others; // the other images holding some of grown
    for (auto const& patch : h.levels.at(ln).patches)
        for (auto const& shift : shifts)
            if (auto const image = core::shift(patch.box, shift);
                image != box and grown * core::grow(image, w))
                others.push_back(image);

    auto const distance = [](Box_t const& b, Point_t const& c) {
        int d = 0;
        for (std::size_t i = 0; i < dim; ++i)
            d += std::max({0, b.lower[i] - c[i], c[i] - b.upper[i]});
        return d;
    };

    std::vector<Box_t> out;
    core::for_each_point(grown, [&](auto const& point) {
        Point_t const c{point};
        if (isIn(c, box))
            return; // distance 0: its own
        auto const mine = distance(box, c);
        for (auto const& o : others)
            if (isIn(c, core::grow(o, w)))
                if (auto const d = distance(o, c);
                    d < mine or (d == mine and core::lexicographic_less(o, box)))
                {
                    // along the last direction: extends the previous box when contiguous
                    if (!out.empty())
                    {
                        auto& last = out.back();
                        bool row   = last.upper[dim - 1] + 1 == c[dim - 1];
                        for (std::size_t i = 0; i + 1 < dim; ++i)
                            row = row and last.lower[i] == c[i];
                        if (row)
                        {
                            last.upper[dim - 1] = c[dim - 1];
                            return;
                        }
                    }
                    out.push_back(Box_t{c, c});
                    return;
                }
    });
    return out;
}

} // namespace damn::amr

#endif // DAMN_AMR_HIERARCHY_HPP
