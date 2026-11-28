#ifndef DAMN_AMR_TOPOLOGY_HPP
#define DAMN_AMR_TOPOLOGY_HPP

// where patches live, and how far one patch's data is from another's
// a Topology with one entry is a single rank; damn/mpi gathers one over ranks

#include "damn/core/def.hpp"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <cstddef>

namespace damn::amr
{
// ordered: code asks "is it at most Local?"
enum class Distance : std::uint8_t {
    Self,   // a periodic image of the same patch
    Local,  // same rank, same memory domain (same NUMA node, same device)
    Numa,   // same rank, another memory domain (another NUMA node, a peer device)
    Node,   // another rank on the same node and memory domain (shared memory)
    Socket, // another rank on the same node, another memory domain (shared memory across them)
    Remote, // another node
};

MKN_KUL_FN_SIG_RET constexpr std::string_view to_string(Distance const d)
{
    switch (d)
    {
        case Distance::Self: return "Self";
        case Distance::Local: return "Local";
        case Distance::Numa: return "Numa";
        case Distance::Node: return "Node";
        case Distance::Socket: return "Socket";
        case Distance::Remote: return "Remote";
    }
    return "?";
}

struct Location
{
    int rank   = 0;
    int node   = 0;  // shared memory domain of ranks
    int numa   = 0;  // memory domain within the node
    int device = -1; // GPU, -1: host memory

    bool operator==(Location const&) const = default;
};

// distance between data in two different patches at a and b
MKN_KUL_FN_SIG_RET constexpr Distance distance(Location const& a, Location const& b)
{
    if (a.rank == b.rank)
        return a.numa == b.numa and a.device == b.device ? Distance::Local : Distance::Numa;
    if (a.node != b.node)
        return Distance::Remote;
    return a.numa == b.numa ? Distance::Node : Distance::Socket;
}

// gathered once, the same on every rank
struct Topology
{
    std::vector<Location> ranks{Location{}}; // ranks[r]: rank r's location
};

// n ranks on one node and one memory domain, e.g. tests and single node runs before a real
// topology is gathered
MKN_KUL_FN_SIG_RET inline Topology single_node(int const n_ranks = 1)
{
    Topology t;
    t.ranks.resize(n_ranks);
    for (int r = 0; r < n_ranks; ++r)
        t.ranks[r].rank = r;
    return t;
}

MKN_KUL_FN_SIG_RET inline Location const& location(Topology const& t, int const rank)
{
    if (rank < 0 or static_cast<std::size_t>(rank) >= t.ranks.size())
        throw std::runtime_error("damn::amr::Topology: no rank " + std::to_string(rank));
    return t.ranks[rank];
}

MKN_KUL_FN_SIG_RET inline Distance distance(Topology const& t, int const rank_a, int const rank_b)
{
    return distance(location(t, rank_a), location(t, rank_b));
}

// an equal share of its node per rank, 1 / (ranks on the node): nodes taken as equal (hwloc or a
// benchmark at init could measure them)
MKN_KUL_FN_SIG_RET inline std::vector<double> node_share_capacity(Topology const& t)
{
    std::map<int, int> per_node; // ranks
    for (auto const& l : t.ranks)
        ++per_node[l.node];
    std::vector<double> capacity;
    capacity.reserve(t.ranks.size());
    for (auto const& l : t.ranks)
        capacity.push_back(1. / per_node[l.node]);
    return capacity;
}

} // namespace damn::amr

#endif // DAMN_AMR_TOPOLOGY_HPP
