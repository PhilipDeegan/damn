#ifndef DAMN_AMR_NEIGHBOURS_HPP
#define DAMN_AMR_NEIGHBOURS_HPP

// Neighbours: the patches of a level near each of a rank's patches, periodic images included
//
//   Link:        another patch (or a periodic image of the same one), the shift to its image
//                and how far its data is (Distance)
//   Neighbours:  per local patch, in LevelData order, every patch whose box grown by reach
//                meets the local patch's box grown by reach: reach is the most cells any
//                resource reaches outside its patch (ghosts, plus one for primal nodes)
//
// purely geometric: built once per regrid (make_neighbours), then damn/amr/copy.hpp copies
// data along the links; what moves where for a given set of resources is planned on first
// use and kept here (plans), so a regrid, which builds new Neighbours, drops them

#include "damn/core/def.hpp"
#include "damn/core/logger.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/topology.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace damn::amr
{
inline constexpr std::size_t not_local = std::numeric_limits<std::size_t>::max();

template<std::size_t dim>
struct Link
{
    Patch<dim> patch{};            // the linked patch
    core::Point<int, dim> shift{}; // its image is at patch.box + shift (linked level's space)
    Distance distance = Distance::Remote;
    std::size_t local = not_local; // its LevelData index, if this rank holds it
};

// what moves between patches for a resource: from src's resource to dst's (the same name for
// fields, e.g. a patch ghost particle array to a domain one for particles)
struct ResourcePair
{
    std::string src{};
    std::string dst{};

    auto operator<=>(ResourcePair const&) const = default;
};

MKN_KUL_FN_SIG_RET inline std::vector<ResourcePair>
same_names(std::vector<std::string> const& names)
{
    std::vector<ResourcePair> pairs;
    pairs.reserve(names.size());
    for (auto const& name : names)
        pairs.push_back({name, name});
    return pairs;
}

// what goes where, see copy.hpp
enum class Region : std::uint8_t {
    Ghosts,  // neighbours' domain points -> my ghost points
    Borders, // neighbours' ghost points -> the same points of mine (ghost boxes overlap)
    Leaving, // neighbours' items outside their domain -> my domain, where their cell is mine
};

// one region of one resource from one linked patch into one patch
template<std::size_t dim>
struct Piece
{
    std::size_t dst = 0; // local patch (LevelData index), not_local for outgoing pieces
    Link<dim> src{};     // where from
    ResourcePair names{};
    core::Box<int, dim> box{}; // dst points (or cells), level index space
    std::size_t offset = 0;    // into the plan's buffer, if it goes through one (bytes)
    std::size_t bytes  = 0;    // field values (0 for particles: their number varies)
    int dst_rank       = 0;
    std::uint32_t pair = 0; // its resources' index in the copy's pairs (and accessors)
};

// the pieces of one destination patch and resource pair, summed together (copy.hpp's fold_sums)
template<std::size_t dim>
struct FoldGroup
{
    std::size_t begin = 0, end = 0;             // its pieces' indices: Plan::fold_order[begin, end)
    std::vector<core::Box<int, dim>> covered{}; // the points they cover, as disjoint boxes
    std::size_t points = 0;                     // in covered
};

// what goes where for a region and set of resources, from one rank's point of view
//   pieces:   into my patches, from mine or another rank's
//   outgoing: from my patches into another rank's
// both ranks of a transfer make the same pieces in the same order, from the same global patch
// list: messages need no description of their content. What follows from the pieces alone is
// made with the plan (copy.hpp's plan_for), so a copy only moves values
template<std::size_t dim>
struct Plan
{
    Region region = Region::Ghosts;
    std::vector<Piece<dim>> pieces{};
    std::vector<Piece<dim>> outgoing{};
    std::vector<std::byte> buffer{}; // local sums/maxes, kept between copies
    std::map<int, std::vector<std::byte>> sent{}, received{}; // per peer rank, kept as well
    std::shared_ptr<void> device{}; // device_copy.hpp's pieces table (GPU builds), kept as well

    // per other rank sending me pieces (rank order): their indices in pieces, in plan order
    std::vector<std::pair<int, std::vector<std::size_t>>> from{};
    // Borders: pieces by destination and resource pair, then their source's image in the
    //  destination's space (fold_sums' order), and the groups of one destination and pair
    std::vector<std::size_t> fold_order{};
    std::vector<FoldGroup<dim>> fold_groups{};
    std::vector<std::byte const*> sources{}; // per piece, its values for fold_sums (per copy)
};

template<std::size_t dim>
struct Neighbours
{
    using PlanKey = std::pair<Region, std::vector<ResourcePair>>;

    int level = 0;
    int rank  = 0;
    int reach = 0;
    std::vector<Patch<dim>> built_on{};     // the level's patches when built
    std::vector<Patch<dim>> local{};        // this rank's, in LevelData order
    std::vector<std::vector<Link<dim>>> of; // of[i]: links of local[i]
    // other ranks' patches near mine, in level order, with their links to my patches: what I
    // send them
    std::vector<std::pair<Patch<dim>, std::vector<Link<dim>>>> remote{};
    std::map<PlanKey, Plan<dim>> plans{}; // copy.hpp's, by region and resources
};


// level ln's neighbours of rank's patches
template<std::size_t dim>
MKN_KUL_FN_SIG_RET Neighbours<dim> make_neighbours(Hierarchy<dim> const& h, int const ln,
                                                   int const rank, Topology const& topology,
                                                   int const reach)
{
    DAMN_LOG_SCOPE(1, "damn::amr::make_neighbours");
    if (!level_exists(h, ln))
        throw std::runtime_error("damn::amr::make_neighbours: no level " + std::to_string(ln));
    if (reach < 0)
        throw std::runtime_error("damn::amr::make_neighbours: negative reach");

    auto const& level = h.levels[ln];

    std::vector<core::Point<int, dim>> shifts{core::Point<int, dim>{}};
    for (auto const& s : periodic_shifts(h, ln))
        shifts.push_back(s);

    Neighbours<dim> n{ln, rank, reach, level.patches, local_patches(level, rank), {}, {}, {}};

    std::map<std::uint32_t, std::size_t> local_index_of;
    for (std::size_t i = 0; i < n.local.size(); ++i)
        local_index_of[n.local[i].id] = i;

    // dst's links to the level's patches (only to rank's if mine_only), in level order: every
    // rank makes the same links for a pair of patches, so their pieces come in the same order
    auto const links_of = [&](Patch<dim> const& dst, bool const mine_only) {
        std::vector<Link<dim>> links;
        auto const reach_of_dst = core::grow(dst.box, reach);
        for (auto const& src : level.patches)
            for (auto const& shift : shifts)
            {
                bool const self = src.id == dst.id;
                if ((self and shift == core::Point<int, dim>{}) or (mine_only and src.rank != rank))
                    continue;
                if (!(reach_of_dst * core::shift(core::grow(src.box, reach), shift)))
                    continue;

                auto const found = local_index_of.find(src.id);
                links.push_back({src, shift,
                                 self ? Distance::Self : distance(topology, dst.rank, src.rank),
                                 found == local_index_of.end() ? not_local : found->second});
            }
        return links;
    };

    for (auto const& dst : n.local)
        n.of.push_back(links_of(dst, false));

    if (!n.local.empty())
        for (auto const& dst : level.patches)
            if (dst.rank != rank)
                if (auto links = links_of(dst, true); !links.empty())
                    n.remote.emplace_back(dst, std::move(links));

    return n;
}

// true if level still has the patches the neighbours were built on
template<std::size_t dim>
MKN_KUL_FN_SIG_RET bool is_current(Neighbours<dim> const& n, Level<dim> const& level)
{
    return n.level == level.number and n.built_on == level.patches;
}

// the farthest any local patch's data has to come from
template<std::size_t dim>
MKN_KUL_FN_SIG_RET Distance farthest(Neighbours<dim> const& n)
{
    Distance d = Distance::Self;
    for (auto const& links : n.of)
        for (auto const& link : links)
            if (link.distance > d)
                d = link.distance;
    return d;
}

} // namespace damn::amr

#endif // DAMN_AMR_NEIGHBOURS_HPP
