#ifndef DAMN_MPI_TOPOLOGY_HPP
#define DAMN_MPI_TOPOLOGY_HPP

// the Topology of comm's ranks: which share a node (MPI_COMM_TYPE_SHARED). One memory domain
// and no device per rank for now (a rank per core)

#include "damn/core/def.hpp"

#include "damn/amr/topology.hpp"

#include "damn/mpi/damn_mpi.hpp"

#include <algorithm>
#include <vector>

namespace damn::mpi
{
MKN_KUL_FN_SIG_RET inline amr::Topology topology(MPI_Comm const comm = MPI_COMM_WORLD)
{
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // a node is named by its lowest rank
    MPI_Comm shared;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    int leader = rank;
    MPI_Allreduce(MPI_IN_PLACE, &leader, 1, MPI_INT, MPI_MIN, shared);
    MPI_Comm_free(&shared);

    std::vector<int> leaders(size);
    MPI_Allgather(&leader, 1, MPI_INT, leaders.data(), 1, MPI_INT, comm);

    auto nodes = leaders;
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());

    amr::Topology t;
    t.ranks.resize(size);
    for (int r = 0; r < size; ++r)
        t.ranks[r] = {r,
                      static_cast<int>(std::lower_bound(nodes.begin(), nodes.end(), leaders[r])
                                       - nodes.begin()),
                      0, -1};
    return t;
}

} // namespace damn::mpi

#endif // DAMN_MPI_TOPOLOGY_HPP
