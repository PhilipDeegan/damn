#ifndef DAMN_MPI_COPY_HPP
#define DAMN_MPI_COPY_HPP

// damn::amr copies between patches of any rank, collective: doc/mpi/copy.md
// every source is read (1, 2) before anything is written (3, 4), so sums and maxes use the
// values from before the copy

#include "damn/core/logger.hpp"
#include "damn/amr/copy.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/mpi/damn_mpi.hpp"
#include "damn/core/memory/scratch.hpp"

#include <cstddef>
#include <limits>
#include <span>
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

struct CopySends; // core::scratch tag: a copy's send requests

// access: amr::by_name, or accessors one per pair
template<amr::Region region, amr::Arrival arrival, std::size_t dim, typename Access,
         typename PatchData>
void copy(amr::CopyOp<region, arrival> const op, amr::Neighbours<dim>& n,
          std::vector<amr::ResourcePair> const& pairs, Access const& access,
          amr::LevelData<dim, PatchData>& ld, MPI_Comm const comm = MPI_COMM_WORLD)
{
    using namespace amr;
    DAMN_LOG_SCOPE(2, "damn::mpi::copy<" + std::string{name(op)} + ">");
    check_access(access, pairs.size());
    auto& plan = plan_for(region, n, ld, pairs);

    auto const as_int = [](std::size_t const bytes) {
        if (bytes > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("damn::mpi::copy: a message over 2GB");
        return static_cast<int>(bytes);
    };

    // 1. send
    std::span<MPI_Request> sends;
    {
        DAMN_LOG_SCOPE(2, "damn::mpi::copy::pack_send");
        for (auto& [_, buffer] : plan.sent)
            buffer.clear();
        for (auto const& piece : plan.outgoing)
        {
            DAMN_LOG_SCOPE(3, "damn::mpi::copy::pack_send::piece"); // one patch overlap
            pack_piece(region, piece, access, ld, plan.sent[piece.dst_rank]);
        }

        sends          = core::scratch<MPI_Request, CopySends>(plan.sent.size());
        std::size_t at = 0;
        for (auto& [peer, buffer] : plan.sent)
            MPI_Isend(buffer.data(), as_int(buffer.size()), MPI_BYTE, peer, copy_tag, comm,
                      &sends[at++]);
    }

    // 2. local
    {
        DAMN_LOG_SCOPE(2, "damn::mpi::copy::local");
        if constexpr (arrival == Arrival::Assign)
            amr::detail::assign(plan, access, ld);
        else if constexpr (arrival == Arrival::Append)
            amr::detail::append_items(plan, access, ld);
        else
            amr::detail::pack_local(plan, access, ld);
    }

    // 3. receive: pieces from each peer, in plan order
    if constexpr (arrival == Arrival::Sum)
        amr::detail::local_sources(plan); // and the received ones below: plan.sources
    for (auto const& [peer, indices] : plan.from)
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
        {
            DAMN_LOG_SCOPE(2, "damn::mpi::copy::recv"); // the probed message out of MPI
            MPI_Recv(buffer.data(), bytes, MPI_BYTE, peer, copy_tag, comm, MPI_STATUS_IGNORE);
        }

        DAMN_LOG_SCOPE(2, "damn::mpi::copy::unpack");
        std::byte const* at = buffer.data();
        for (auto const i : indices)
        {
            DAMN_LOG_SCOPE(3, "damn::mpi::copy::unpack::piece"); // one patch overlap
            auto const& piece = plan.pieces[i];
            if constexpr (arrival == Arrival::Sum)
            {
                if (at + piece.bytes > buffer.data() + buffer.size())
                    break; // checked below
                plan.sources[i] = at;
                at += piece.bytes;
            }
            else
                unpack_piece<arrival>(region, piece, access, ld, at);
        }
        if (at != buffer.data() + buffer.size())
            throw std::runtime_error("damn::mpi::copy: rank " + std::to_string(peer)
                                     + "'s message does not match this rank's plan");
    }

    // 4. finish
    {
        DAMN_LOG_SCOPE(2, "damn::mpi::copy::finish");
        if constexpr (arrival == Arrival::Sum)
            amr::detail::fold_sums(plan, access, ld);
        else if constexpr (arrival == Arrival::Max)
            amr::detail::unpack_local<arrival>(plan, access, ld);
        else if constexpr (arrival == Arrival::Append) // local and received items in, each once
            amr::detail::appended(plan, access, ld);
    }
    DAMN_LOG_SCOPE(2, "damn::mpi::copy::sends"); // until the peers have received them
    MPI_Waitall(static_cast<int>(sends.size()), sends.data(), MPI_STATUSES_IGNORE);
}

template<amr::Region region, amr::Arrival arrival, std::size_t dim, typename PatchData>
void copy(amr::CopyOp<region, arrival> const op, amr::Neighbours<dim>& n,
          std::vector<amr::ResourcePair> const& pairs, amr::LevelData<dim, PatchData>& ld,
          MPI_Comm const comm = MPI_COMM_WORLD)
{
    copy(op, n, pairs, amr::by_name, ld, comm);
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
