#ifndef DAMN_MPI_COPY_HPP
#define DAMN_MPI_COPY_HPP

// damn::amr copies over MPI: the same copies as damn/amr/copy.hpp's, between patches of any
// rank, collective over the ranks holding the level's patches
//
//   damn::mpi::copy(damn::amr::fill_ghosts, neighbours, {"Bx"}, level_data)
//
// every rank plans the same pieces from the same global patch list, so a message is the
// pieces' data only, in plan order: one message per peer rank and copy
//   1. outgoing pieces packed per peer and sent (MPI_Isend)
//   2. local pieces: ghosts and leaving items copied, sums and maxes read into the buffer
//   3. per peer with pieces for me: probed (particle messages vary in size), received,
//      unpacked; sums kept in the received buffers
//   4. local maxes written, sums folded with the received ones in a fixed order
//      (damn::amr::detail::fold_sums), sends waited for
// every source is read (1, 2) before anything is written (3, 4), so sums and maxes use the
// values from before the copy

#include "damn/core/logger.hpp"
#include "damn/amr/copy.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/mpi/damn_mpi.hpp"

#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace damn::amr
{
template<std::size_t dim, typename PatchData>
struct LevelData;
} // namespace damn::amr

namespace damn::mpi
{
inline constexpr int copy_tag = 0xda;

template<amr::Region region, amr::Arrival arrival, std::size_t dim, typename PatchData>
void copy(amr::CopyOp<region, arrival> const op, amr::Neighbours<dim>& n,
          std::vector<amr::ResourcePair> const& pairs, amr::LevelData<dim, PatchData>& ld,
          MPI_Comm const comm = MPI_COMM_WORLD)
{
    using namespace amr;
    DAMN_LOG_SCOPE(2, "damn::mpi::copy<" + std::string{name(op)} + ">");
    auto& plan = plan_for(region, n, ld, pairs);

    auto const as_int = [](std::size_t const bytes) {
        if (bytes > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("damn::mpi::copy: a message over 2GB");
        return static_cast<int>(bytes);
    };

    // 1. send
    std::vector<MPI_Request> sends;
    {
        DAMN_LOG_SCOPE(2, "damn::mpi::copy::pack_send");
        for (auto& [_, buffer] : plan.sent)
            buffer.clear();
        for (auto const& piece : plan.outgoing)
            pack_piece(region, piece, ld, plan.sent[piece.dst_rank]);

        sends.reserve(plan.sent.size());
        for (auto& [peer, buffer] : plan.sent)
            MPI_Isend(buffer.data(), as_int(buffer.size()), MPI_BYTE, peer, copy_tag, comm,
                      &sends.emplace_back());
    }

    // 2. local
    {
        DAMN_LOG_SCOPE(2, "damn::mpi::copy::local");
        if constexpr (arrival == Arrival::Assign)
            amr::detail::assign(plan, ld);
        else if constexpr (arrival == Arrival::Append)
            amr::detail::append_items(plan, ld);
        else
            amr::detail::pack_local(plan, ld);
    }

    // 3. receive: pieces from each peer, in plan order
    std::vector<std::byte const*> sources; // sums: each piece's values, local or received
    if constexpr (arrival == Arrival::Sum)
        sources = amr::detail::local_sources(plan);
    std::map<int, std::vector<Piece<dim> const*>> from;
    for (auto const& piece : plan.pieces)
        if (piece.src.local == not_local)
            from[piece.src.patch.rank].push_back(&piece);

    for (auto const& [peer, pieces] : from)
    {
        MPI_Status status;
        {
            DAMN_LOG_SCOPE(2, "damn::mpi::copy::wait"); // for the peer's message
            MPI_Probe(peer, copy_tag, comm, &status);
        }
        int bytes = 0;
        MPI_Get_count(&status, MPI_BYTE, &bytes);
        auto& buffer = plan.received[peer];
        buffer.resize(static_cast<std::size_t>(bytes));
        MPI_Recv(buffer.data(), bytes, MPI_BYTE, peer, copy_tag, comm, MPI_STATUS_IGNORE);

        DAMN_LOG_SCOPE(2, "damn::mpi::copy::unpack");
        std::byte const* at = buffer.data();
        for (auto const* piece : pieces)
            if constexpr (arrival == Arrival::Sum)
            {
                if (at + piece->bytes > buffer.data() + buffer.size())
                    break; // checked below
                sources[static_cast<std::size_t>(piece - plan.pieces.data())] = at;
                at += piece->bytes;
            }
            else
                unpack_piece<arrival>(region, *piece, ld, at);
        if (at != buffer.data() + buffer.size())
            throw std::runtime_error("damn::mpi::copy: rank " + std::to_string(peer)
                                     + "'s message does not match this rank's plan");
    }

    // 4.
    DAMN_LOG_SCOPE(2, "damn::mpi::copy::finish");
    if constexpr (arrival == Arrival::Sum)
        amr::detail::fold_sums(plan, ld, sources);
    else if constexpr (arrival == Arrival::Max)
        amr::detail::unpack_local<arrival>(plan, ld);
    else if constexpr (arrival == Arrival::Append) // local and received items in, each once
        amr::detail::appended(plan, ld);
    MPI_Waitall(static_cast<int>(sends.size()), sends.data(), MPI_STATUSES_IGNORE);
}

// fields: the same name on both sides
template<amr::Region region, amr::Arrival arrival, std::size_t dim, typename PatchData>
void copy(amr::CopyOp<region, arrival> const op, amr::Neighbours<dim>& n,
          std::vector<std::string> const& names, amr::LevelData<dim, PatchData>& ld,
          MPI_Comm const comm = MPI_COMM_WORLD)
{
    copy(op, n, amr::same_names(names), ld, comm);
}

} // namespace damn::mpi

#endif // DAMN_MPI_COPY_HPP
