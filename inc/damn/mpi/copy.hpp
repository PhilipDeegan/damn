#ifndef DAMN_MPI_COPY_HPP
#define DAMN_MPI_COPY_HPP

// damn::amr copies between patches of any rank, collective: doc/mpi/copy.md
// every source is read (1, 2) before anything is written (3, 4), so sums and maxes use the
// values from before the copy. Device resources take a device path: doc/mpi/copy.md

#include "damn/core/logger.hpp"
#include "damn/amr/copy.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/mpi/damn_mpi.hpp"
#include "damn/core/memory/scratch.hpp"
#include "damn/core/def/damn_config.hpp"

#if DAMN_HAVE_GPU && __has_include(<mpi-ext.h>)
#include <mpi-ext.h> // MPIX_Query_cuda_support, MPIX_Query_rocm_support (Open MPI)
#endif

#include <cstddef>
#include <limits>
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
inline constexpr int copy_tag        = 0xda;
inline constexpr int copy_counts_tag = 0xdd; // device items: their counts before them

namespace detail
{
    inline int as_count(std::size_t const bytes)
    {
        if (bytes > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("damn::mpi::copy: a message over 2GB");
        return static_cast<int>(bytes);
    }

#if DAMN_HAVE_GPU
    // device buffers handed to MPI: an MPI that does not read them on the device would read
    //  them on the host
    inline void check_device_aware()
    {
        static bool const aware = [] {
#if MKN_GPU_CUDA && defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
            return MPIX_Query_cuda_support() == 1;
#elif MKN_GPU_ROCM && defined(MPIX_ROCM_AWARE_SUPPORT) && MPIX_ROCM_AWARE_SUPPORT
            return MPIX_Query_rocm_support() == 1;
#else
            return false;
#endif
        }();
        if (!aware)
            throw std::runtime_error("damn::mpi::copy: device resources need a GPU aware MPI");
    }

    // the device path: field messages' sizes are the plan's, so receives are posted first
    template<amr::Arrival arrival, std::size_t dim, typename PatchData>
    void copy_on_device(amr::Plan<dim> const& plan, amr::detail::DeviceTable<dim>& table,
                        amr::LevelData<dim, PatchData> const& ld, MPI_Comm const comm)
    {
        if (!table.to.empty() or !table.from.empty())
            check_device_aware();
        auto const value = sizeof(double);

        core::ScratchVector<MPI_Request> recvs;
        recvs.resize(table.from.size());
        for (std::size_t k = 0; k < table.from.size(); ++k)
        {
            auto const& peer = table.from[k];
            MPI_Irecv(table.recv.data() + peer.begin, as_count(peer.count * value), MPI_BYTE,
                      peer.rank, copy_tag, comm, &recvs[k]);
        }

        core::ScratchVector<MPI_Request> sends;
        sends.resize(table.to.size());
        {
            DAMN_LOG_SCOPE(2, "damn::mpi::copy::pack_send");
            amr::pack_on_device(table); // done when it returns: the launch waits
            for (std::size_t k = 0; k < table.to.size(); ++k)
            {
                auto const& peer = table.to[k];
                MPI_Isend(table.send.data() + peer.begin, as_count(peer.count * value), MPI_BYTE,
                          peer.rank, copy_tag, comm, &sends[k]);
            }
        }

        core::ScratchVector<MPI_Status> status;
        status.resize(table.from.size());
        {
            DAMN_LOG_SCOPE(2, "damn::mpi::copy::wait"); // for the peers' messages
            MPI_Waitall(static_cast<int>(recvs.size()), recvs.data(), status.data());
        }
        for (std::size_t k = 0; k < table.from.size(); ++k)
        {
            int bytes = 0;
            MPI_Get_count(&status[k], MPI_BYTE, &bytes);
            if (static_cast<std::size_t>(bytes) != table.from[k].count * value)
                throw std::runtime_error("damn::mpi::copy: rank "
                                         + std::to_string(table.from[k].rank)
                                         + "'s message does not match this rank's plan");
        }

        {
            DAMN_LOG_SCOPE(2, "damn::mpi::copy::finish");
            amr::finish_on_device<arrival>(plan, table, ld);
        }
        DAMN_LOG_SCOPE(2, "damn::mpi::copy::sends"); // until the peers have received them
        MPI_Waitall(static_cast<int>(sends.size()), sends.data(), MPI_STATUSES_IGNORE);
    }
#endif // DAMN_HAVE_GPU

#if DAMN_HAVE_GPU && DAMN_HAVE_THRUST
    // leaving items on the device: per peer a counts message (host, its size the plan's), then
    //  its blocks (device), received at the place the counts give
    template<std::size_t dim, typename Access, typename PatchData>
    void items_on_device(amr::Plan<dim> const& plan, amr::detail::ItemTable<dim>& table,
                         Access const& access, amr::LevelData<dim, PatchData>& ld,
                         MPI_Comm const comm)
    {
        if (!table.to.empty() or !table.from.empty())
            check_device_aware();
        auto const words = sizeof(std::uint64_t);

        core::ScratchVector<MPI_Request> counts;
        counts.resize(table.from.size());
        for (std::size_t k = 0; k < table.from.size(); ++k)
        {
            auto& peer = table.from[k];
            MPI_Irecv(peer.header.data(), as_count(peer.header.size() * words), MPI_BYTE, peer.rank,
                      copy_counts_tag, comm, &counts[k]);
        }

        core::ScratchVector<MPI_Request> sends;
        sends.resize(2 * table.to.size());
        {
            DAMN_LOG_SCOPE(2, "damn::mpi::copy::pack_send");
            amr::pack_items_on_device(plan, access, ld, table); // done when it returns
            auto const* const send = reinterpret_cast<std::byte const*>(table.send.data());
            for (std::size_t k = 0; k < table.to.size(); ++k)
            {
                auto const& peer = table.to[k];
                MPI_Isend(peer.header.data(), as_count(peer.header.size() * words), MPI_BYTE,
                          peer.rank, copy_counts_tag, comm, &sends[2 * k]);
                MPI_Isend(send + peer.begin, as_count(peer.bytes), MPI_BYTE, peer.rank, copy_tag,
                          comm, &sends[2 * k + 1]);
            }
        }

        {
            DAMN_LOG_SCOPE(2, "damn::mpi::copy::wait"); // for the peers' counts
            MPI_Waitall(static_cast<int>(counts.size()), counts.data(), MPI_STATUSES_IGNORE);
        }
        std::size_t bytes = 0;
        for (std::size_t k = 0; k < table.from.size(); ++k)
        {
            auto& peer = table.from[k];
            peer.begin = bytes;
            peer.bytes = amr::detail::received_bytes(plan, table, k);
            bytes += peer.bytes;
        }
        table.recv.resize(bytes / words);

        core::ScratchVector<MPI_Request> recvs;
        recvs.resize(table.from.size());
        core::ScratchVector<MPI_Status> status;
        status.resize(table.from.size());
        auto* const recv = reinterpret_cast<std::byte*>(table.recv.data());
        for (std::size_t k = 0; k < table.from.size(); ++k)
        {
            auto const& peer = table.from[k];
            MPI_Irecv(recv + peer.begin, as_count(peer.bytes), MPI_BYTE, peer.rank, copy_tag, comm,
                      &recvs[k]);
        }
        {
            DAMN_LOG_SCOPE(2, "damn::mpi::copy::wait"); // for the peers' items
            MPI_Waitall(static_cast<int>(recvs.size()), recvs.data(), status.data());
        }
        for (std::size_t k = 0; k < table.from.size(); ++k)
        {
            int got = 0;
            MPI_Get_count(&status[k], MPI_BYTE, &got);
            if (static_cast<std::size_t>(got) != table.from[k].bytes
                or !amr::place_received(plan, table, k))
                throw std::runtime_error("damn::mpi::copy: rank "
                                         + std::to_string(table.from[k].rank)
                                         + "'s items do not match its counts");
        }

        {
            DAMN_LOG_SCOPE(2, "damn::mpi::copy::finish");
            amr::unpack_items_on_device(plan, access, ld, table);
            amr::detail::appended(plan, access, ld); // local and received items in, each once
        }
        DAMN_LOG_SCOPE(2, "damn::mpi::copy::sends"); // until the peers have received them
        MPI_Waitall(static_cast<int>(sends.size()), sends.data(), MPI_STATUSES_IGNORE);
    }
#endif // DAMN_HAVE_GPU && DAMN_HAVE_THRUST
} // namespace detail

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

#if DAMN_HAVE_GPU
    if constexpr (region != Region::Leaving)
        if (auto& table = amr::detail::device_table(plan, access, ld); table.usable)
            return detail::copy_on_device<arrival>(plan, table, ld, comm);
#endif
#if DAMN_HAVE_GPU && DAMN_HAVE_THRUST
    if constexpr (region == Region::Leaving)
        if (auto& table = amr::detail::item_table(plan, access, ld); table.usable)
            return detail::items_on_device(plan, table, access, ld, comm);
#endif

    // 1. send
    core::ScratchVector<MPI_Request> sends; // until the sends are waited for, step 4
    {
        DAMN_LOG_SCOPE(2, "damn::mpi::copy::pack_send");
        for (auto& [_, buffer] : plan.sent)
            buffer.clear();
        for (auto const& piece : plan.outgoing)
        {
            DAMN_LOG_SCOPE(3, "damn::mpi::copy::pack_send::piece"); // one patch overlap
            pack_piece(region, piece, access, ld, plan.sent[piece.dst_rank]);
        }

        sends.resize(plan.sent.size());
        std::size_t at = 0;
        for (auto& [peer, buffer] : plan.sent)
            MPI_Isend(buffer.data(), detail::as_count(buffer.size()), MPI_BYTE, peer, copy_tag,
                      comm, &sends[at++]);
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
