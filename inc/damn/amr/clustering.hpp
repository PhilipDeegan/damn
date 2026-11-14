#ifndef DAMN_AMR_CLUSTERING_HPP
#define DAMN_AMR_CLUSTERING_HPP

// the boxes of a new level ln + 1 from the cells of level ln tagged for refinement, by tiles
// (as SAMRAI's TileClustering):
//
//   1. each tagged cell is grown by tag_buffer cells
//   2. the tiles of the grown cells: the level's cells cut in tile_size boxes from cell 0,
//      periodic images wrapped into the domain, clipped to it
//   3. each tile keeps its cells properly nested in level ln: nesting_buffer cells inside level
//      ln's patches (or a physical, non periodic, domain boundary)
//   4. the tiles left are coalesced: two boxes whose union is a box become one, until none do
//   5. refined into level ln + 1's index space
//
// tagged_tiles (1, 2) is per rank; cluster (3, 4, 5) works on every rank's tiles, so every rank
// gets the same boxes: damn/mpi/clustering.hpp gathers them over the ranks

#include "damn/core/def.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/core/math/array.hpp"
#include "damn/amr/hierarchy.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace damn::amr
{
template<std::size_t dim>
struct ClusteringOptions
{
    int tag_buffer                 = 1;                             // level ln cells
    int nesting_buffer             = 1;                             // level ln cells
    std::array<int, dim> tile_size = core::ConstArray<int, dim>(8); // level ln cells
};

// a tile: its index on the tile grid of level ln (std::array: ordered)
template<std::size_t dim>
using Tile = std::array<int, dim>;


namespace detail
{
    // the cell wrapped into the level's domain where periodic, std::nullopt if outside it
    template<std::size_t dim>
    NO_DISCARD std::optional<core::Point<int, dim>> wrap(Hierarchy<dim> const& h, int const ln,
                                                         core::Point<int, dim> cell)
    {
        auto const domain = level_domain(h, ln);
        auto const shape  = domain.shape();
        for (std::size_t i = 0; i < dim; ++i)
        {
            if (h.periodic[i])
                cell[i] = domain.lower[i]
                          + ((cell[i] - domain.lower[i]) % shape[i] + shape[i]) % shape[i];
            else if (cell[i] < domain.lower[i] or cell[i] > domain.upper[i])
                return std::nullopt;
        }
        return cell;
    }

    template<std::size_t dim>
    NO_DISCARD core::Box<int, dim> tile_box(Hierarchy<dim> const& h, int const ln,
                                            Tile<dim> const& tile, std::array<int, dim> const& size)
    {
        core::Box<int, dim> box;
        for (std::size_t i = 0; i < dim; ++i)
        {
            box.lower[i] = tile[i] * size[i];
            box.upper[i] = box.lower[i] + size[i] - 1;
        }
        auto const clipped = box * level_domain(h, ln);
        if (!clipped)
            throw std::runtime_error("damn::amr::cluster: a tile outside the domain");
        return *clipped;
    }

    // true if cell is properly nested in level ln: every cell within nesting_buffer of it is on
    // a patch of level ln (or past a non periodic domain boundary)
    template<std::size_t dim>
    NO_DISCARD bool nested(Hierarchy<dim> const& h, int const ln, core::Point<int, dim> const& cell,
                           int const buffer)
    {
        auto around       = core::grow(core::Box<int, dim>{cell, cell}, buffer);
        auto const domain = level_domain(h, ln);
        for (std::size_t i = 0; i < dim; ++i)
            if (!h.periodic[i])
            {
                around.lower[i] = std::max(around.lower[i], domain.lower[i]);
                around.upper[i] = std::min(around.upper[i], domain.upper[i]);
            }
        std::size_t covered = 0;
        for (auto const& b : covered_by_level(h, ln, around))
            covered += b.size();
        return covered >= around.size(); // patches of a level do not overlap
    }

    template<std::size_t dim>
    NO_DISCARD Tile<dim> tile_of(core::Point<int, dim> const& cell,
                                 std::array<int, dim> const& size)
    {
        Tile<dim> tile;
        for (std::size_t i = 0; i < dim; ++i)
            tile[i] = core::floor_div(cell[i], size[i]);
        return tile;
    }

    // a and b as one box, if their union is one
    template<std::size_t dim>
    NO_DISCARD std::optional<core::Box<int, dim>> union_box(core::Box<int, dim> const& a,
                                                            core::Box<int, dim> const& b)
    {
        std::size_t differ = dim;
        for (std::size_t i = 0; i < dim; ++i)
            if (a.lower[i] != b.lower[i] or a.upper[i] != b.upper[i])
            {
                if (differ != dim)
                    return std::nullopt; // more than one direction
                differ = i;
            }
        if (differ == dim)
            return a;
        auto const i = differ;
        if (a.upper[i] + 1 != b.lower[i] and b.upper[i] + 1 != a.lower[i])
            return std::nullopt; // not adjacent
        return a.merge(b);
    }
} // namespace detail


// the tiles holding this rank's tagged cells (level ln), grown by the tag buffer
template<std::size_t dim>
NO_DISCARD std::set<Tile<dim>> tagged_tiles(Hierarchy<dim> const& h, int const ln,
                                            std::vector<core::Point<int, dim>> const& tagged,
                                            ClusteringOptions<dim> const& options)
{
    std::set<Tile<dim>> tiles;
    for (auto const& cell : tagged)
        core::for_each_point(core::grow(core::Box<int, dim>{cell, cell}, options.tag_buffer),
                             [&](auto const& c) {
                                 if (auto const wrapped = detail::wrap(h, ln, c))
                                     tiles.insert(detail::tile_of(*wrapped, options.tile_size));
                             });
    return tiles;
}

// level ln + 1's boxes (its index space) from every rank's tagged tiles of level ln
template<std::size_t dim>
NO_DISCARD std::vector<core::Box<int, dim>> cluster(Hierarchy<dim> const& h, int const ln,
                                                    std::set<Tile<dim>> const& tiles,
                                                    ClusteringOptions<dim> const& options)
{
    if (!level_exists(h, ln))
        throw std::runtime_error("damn::amr::cluster: no level " + std::to_string(ln));

    // the nested part of each tile, as boxes: per tile, its nested cells
    std::vector<core::Box<int, dim>> boxes;
    for (auto const& tile : tiles)
    {
        auto const box = detail::tile_box(h, ln, tile, options.tile_size);
        std::vector<core::Box<int, dim>> nested;
        core::for_each_point(box, [&](auto const& cell) {
            if (detail::nested(h, ln, cell, options.nesting_buffer))
                nested.emplace_back(cell, cell);
        });
        if (nested.size() == box.size())
            boxes.push_back(box);
        else
            boxes.insert(boxes.end(), nested.begin(), nested.end());
    }

    // coalesce until nothing merges: deterministic, in lower corner order
    bool merged = true;
    while (merged)
    {
        merged = false;
        std::sort(boxes.begin(), boxes.end(), [](auto const& a, auto const& b) {
            auto const al = a.lower.toArray(), bl = b.lower.toArray();
            return al < bl or (al == bl and a.upper.toArray() < b.upper.toArray());
        });
        for (std::size_t i = 0; i < boxes.size() and !merged; ++i)
            for (std::size_t j = i + 1; j < boxes.size(); ++j)
                if (auto const u = detail::union_box(boxes[i], boxes[j]))
                {
                    boxes[i] = *u;
                    boxes.erase(boxes.begin() + static_cast<std::ptrdiff_t>(j));
                    merged = true;
                    break;
                }
    }

    for (auto& box : boxes)
        box = core::refine(box, h.ratio);
    return boxes;
}

} // namespace damn::amr

#endif // DAMN_AMR_CLUSTERING_HPP
