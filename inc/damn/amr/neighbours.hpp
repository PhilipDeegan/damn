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
// purely geometric: built once per regrid (make_neighbours), then damn/amr/move.hpp moves
// data along the links; what moves where for a given set of resources is planned on first
// use and kept here (plans), so a regrid, which builds new Neighbours, drops them

#include "damn/core/def.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/amr/topology.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
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

NO_DISCARD inline std::vector<ResourcePair> same_names(std::vector<std::string> const& names)
{
    std::vector<ResourcePair> pairs;
    pairs.reserve(names.size());
    for (auto const& name : names)
        pairs.push_back({name, name});
    return pairs;
}

// what goes where, see move.hpp
enum class Region : std::uint8_t {
    Ghosts,  // neighbours' domain points -> my ghost points
    Borders, // neighbours' ghost points -> the same points of mine (ghost boxes overlap)
    Leaving, // neighbours' items outside their domain -> my domain, where their cell is mine
};

// one region of one resource from one linked patch into one local patch
template<std::size_t dim>
struct Copy
{
    std::size_t dst = 0; // local patch (LevelData index)
    Link<dim> src{};     // where from
    ResourcePair names{};
    core::Box<int, dim> box{}; // dst points (or cells), level index space
    std::size_t offset = 0;    // into the plan's buffer, if it goes through one (bytes)
    std::size_t bytes  = 0;
};

template<std::size_t dim>
struct Plan
{
    std::vector<Copy<dim>> copies{};
    std::vector<std::byte> buffer{}; // kept between moves: no allocation once sized
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
    std::map<PlanKey, Plan<dim>> plans{};   // move.hpp's, by region and resources
};


// level ln's neighbours of rank's patches
template<std::size_t dim>
NO_DISCARD Neighbours<dim> make_neighbours(Hierarchy<dim> const& h, int const ln, int const rank,
                                           Topology const& topology, int const reach)
{
    if (!level_exists(h, ln))
        throw std::runtime_error("damn::amr::make_neighbours: no level " + std::to_string(ln));
    if (reach < 0)
        throw std::runtime_error("damn::amr::make_neighbours: negative reach");

    auto const& level = h.levels[ln];

    std::vector<core::Point<int, dim>> shifts{core::Point<int, dim>{}};
    for (auto const& s : periodic_shifts(h, ln))
        shifts.push_back(s);

    Neighbours<dim> n{ln, rank, reach, level.patches, local_patches(level, rank), {}, {}};

    std::map<std::uint32_t, std::size_t> local_index_of;
    for (std::size_t i = 0; i < n.local.size(); ++i)
        local_index_of[n.local[i].id] = i;

    n.of.resize(n.local.size());
    for (std::size_t i = 0; i < n.local.size(); ++i)
    {
        auto const& dst         = n.local[i];
        auto const reach_of_dst = core::grow(dst.box, reach);

        for (auto const& src : level.patches)
            for (auto const& shift : shifts)
            {
                bool const self = src.id == dst.id;
                if (self and shift == core::Point<int, dim>{})
                    continue;
                if (!(reach_of_dst * core::shift(core::grow(src.box, reach), shift)))
                    continue;

                auto const found = local_index_of.find(src.id);
                n.of[i].push_back({src, shift,
                                   self ? Distance::Self : distance(topology, dst.rank, src.rank),
                                   found == local_index_of.end() ? not_local : found->second});
            }
    }
    return n;
}

// true if level still has the patches the neighbours were built on
template<std::size_t dim>
NO_DISCARD bool is_current(Neighbours<dim> const& n, Level<dim> const& level)
{
    return n.level == level.number and n.built_on == level.patches;
}

// the farthest any local patch's data has to come from
template<std::size_t dim>
NO_DISCARD Distance farthest(Neighbours<dim> const& n)
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
