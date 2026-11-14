#ifndef DAMN_MPI_CLUSTERING_HPP
#define DAMN_MPI_CLUSTERING_HPP

// damn::amr::cluster over MPI: every rank's tagged tiles gathered on every rank, then clustered
// there, so every rank gets the same boxes (collective)
//
//   damn::mpi::cluster(h, ln, my_tagged_cells, options)

#include "damn/amr/clustering.hpp"
#include "damn/mpi/damn_mpi.hpp"

#include <cstddef>
#include <numeric>
#include <set>
#include <vector>

namespace damn::mpi
{
template<std::size_t dim>
NO_DISCARD std::vector<core::Box<int, dim>>
cluster(amr::Hierarchy<dim> const& h, int const ln,
        std::vector<core::Point<int, dim>> const& tagged,
        amr::ClusteringOptions<dim> const& options, MPI_Comm const comm = MPI_COMM_WORLD)
{
    auto const mine = amr::tagged_tiles(h, ln, tagged, options);

    std::vector<int> flat;
    flat.reserve(mine.size() * dim);
    for (auto const& tile : mine)
        flat.insert(flat.end(), tile.begin(), tile.end());

    int n_ranks = 1;
    MPI_Comm_size(comm, &n_ranks);
    int const n = static_cast<int>(flat.size());
    std::vector<int> counts(static_cast<std::size_t>(n_ranks)), displs(counts.size());
    MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    std::partial_sum(counts.begin(), counts.end() - 1, displs.begin() + 1);
    std::vector<int> all(static_cast<std::size_t>(displs.back() + counts.back()));
    MPI_Allgatherv(flat.data(), n, MPI_INT, all.data(), counts.data(), displs.data(), MPI_INT,
                   comm);

    std::set<amr::Tile<dim>> tiles;
    for (std::size_t i = 0; i < all.size(); i += dim)
    {
        amr::Tile<dim> tile;
        for (std::size_t d = 0; d < dim; ++d)
            tile[d] = all[i + d];
        tiles.insert(tile);
    }
    return amr::cluster(h, ln, tiles, options);
}

} // namespace damn::mpi

#endif // DAMN_MPI_CLUSTERING_HPP
