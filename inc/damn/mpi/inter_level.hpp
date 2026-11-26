#ifndef DAMN_MPI_INTER_LEVEL_HPP
#define DAMN_MPI_INTER_LEVEL_HPP

// damn::amr::refine and coarsen between patches of any rank, collective: doc/mpi/inter_level.md

#include "damn/core/logger.hpp"
#include "damn/amr/inter_level.hpp"
#include "damn/amr/level_map.hpp"
#include "damn/mpi/damn_mpi.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/core/def.hpp"
#include "damn/amr/copy.hpp"

#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace damn::mpi
{
inline constexpr int inter_level_tag = 0xdc; // copy_tag, regrid_tag: each transfer its own

namespace detail
{
    MKN_KUL_FN_SIG_RET inline int as_int(std::size_t const bytes)
    {
        if (bytes > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("damn::mpi::inter_level: a message over 2GB");
        return static_cast<int>(bytes);
    }

    // the plan's per peer buffers sent, the requests returned
    MKN_KUL_FN_SIG_RET inline std::vector<MPI_Request>
    send_all(std::map<int, std::vector<std::byte>>& sent, MPI_Comm const comm)
    {
        std::vector<MPI_Request> sends;
        sends.reserve(sent.size());
        for (auto& [peer, buffer] : sent)
            MPI_Isend(buffer.data(), as_int(buffer.size()), MPI_BYTE, peer, inter_level_tag, comm,
                      &sends.emplace_back());
        return sends;
    }

    // per peer: its message received, then unpack(piece, at) for its pieces in plan order
    template<typename Piece>
    void receive_all(std::map<int, std::vector<Piece const*>> const& from,
                     std::map<int, std::vector<std::byte>>& received, MPI_Comm const comm,
                     auto&& unpack)
    {
        for (auto const& [peer, pieces] : from)
        {
            MPI_Status status;
            {
                DAMN_LOG_SCOPE(2, "damn::mpi::inter_level::wait"); // for the peer's message
                MPI_Probe(peer, inter_level_tag, comm, &status);
            }
            int bytes = 0;
            MPI_Get_count(&status, MPI_BYTE, &bytes);
            auto& buffer = received[peer];
            buffer.resize(static_cast<std::size_t>(bytes));
            MPI_Recv(buffer.data(), bytes, MPI_BYTE, peer, inter_level_tag, comm,
                     MPI_STATUS_IGNORE);

            std::byte const* at = buffer.data();
            for (auto const* piece : pieces)
                unpack(*piece, at);
            if (at != buffer.data() + buffer.size())
                throw std::runtime_error("damn::mpi::inter_level: rank " + std::to_string(peer)
                                         + "'s message does not match this rank's plan");
        }
    }
} // namespace detail


// access: amr::by_name, or accessors one per pair
template<std::size_t dim, typename Access, typename CoarseLD, typename FineLD>
    requires amr::LevelDataLike<FineLD>
void refine(auto op, amr::LevelRegion const region, amr::LevelMap<dim>& map,
            std::vector<amr::LevelPair> const& pairs, Access const& access, CoarseLD& coarse_ld,
            FineLD& fine_ld, double const alpha = 0, MPI_Comm const comm = MPI_COMM_WORLD)
{
    using namespace amr;
    DAMN_LOG_SCOPE(2, "damn::mpi::refine");
    if (region == LevelRegion::Coarsened)
        throw std::runtime_error("damn::mpi::refine: not a refine region");
    check_access(access, pairs.size());
    auto& plan = level_plan_for(region, op.stencil, map, pairs, coarse_ld, fine_ld);
    clear_scratch(plan, access, fine_ld);

    auto const fine_rank = [&](auto const& piece) { return plan.targets[piece.target].fine.rank; };

    // coarse data to other ranks' fine patches
    for (auto& [_, buffer] : plan.sent)
        buffer.clear();
    for (auto const& piece : plan.pieces)
        if (amr::detail::coarse_is_local(piece) and !amr::detail::fine_is_local(plan, piece))
            pack_coarse_piece(plan, piece, access, coarse_ld, alpha, plan.sent[fine_rank(piece)]);
    auto sends = detail::send_all(plan.sent, comm);

    // local
    std::vector<std::byte> bytes;
    std::map<int, std::vector<LevelPiece<dim> const*>> from;
    for (auto const& piece : plan.pieces)
    {
        if (!amr::detail::fine_is_local(plan, piece))
            continue;
        if (!amr::detail::coarse_is_local(piece))
        {
            from[piece.coarse.patch.rank].push_back(&piece);
            continue;
        }
        bytes.clear();
        pack_coarse_piece(plan, piece, access, coarse_ld, alpha, bytes);
        std::byte const* at = bytes.data();
        unpack_coarse_piece(plan, piece, access, fine_ld, op, at);
    }

    detail::receive_all(from, plan.received, comm, [&](auto const& piece, auto& at) {
        unpack_coarse_piece(plan, piece, access, fine_ld, op, at);
    });

    refine_fields(plan, access, fine_ld, op);
    MPI_Waitall(static_cast<int>(sends.size()), sends.data(), MPI_STATUSES_IGNORE);
}

template<std::size_t dim, typename CoarseLD, typename FineLD>
    requires amr::LevelDataLike<FineLD>
void refine(auto op, amr::LevelRegion const region, amr::LevelMap<dim>& map,
            std::vector<amr::LevelPair> const& pairs, CoarseLD& coarse_ld, FineLD& fine_ld,
            double const alpha = 0, MPI_Comm const comm = MPI_COMM_WORLD)
{
    refine(op, region, map, pairs, amr::by_name, coarse_ld, fine_ld, alpha, comm);
}

template<std::size_t dim, typename Access, typename FineLD, typename CoarseLD>
    requires amr::LevelDataLike<CoarseLD>
void coarsen(auto op, amr::LevelMap<dim>& map, std::vector<amr::LevelPair> const& pairs,
             Access const& access, FineLD& fine_ld, CoarseLD& coarse_ld,
             MPI_Comm const comm = MPI_COMM_WORLD)
{
    using namespace amr;
    DAMN_LOG_SCOPE(2, "damn::mpi::coarsen");
    check_access(access, pairs.size());
    auto& plan = level_plan_for(LevelRegion::Coarsened, 0, map, pairs, coarse_ld, fine_ld);

    // computed on my fine patches: assigned, or sent to the coarse patch's rank
    for (auto& [_, buffer] : plan.sent)
        buffer.clear();
    std::vector<std::byte> bytes;
    std::map<int, std::vector<LevelPiece<dim> const*>> from;
    for (auto const& piece : plan.pieces) // one per target
    {
        auto const& target = plan.targets[piece.target];
        if (target.fine_local == not_local)
        {
            if (amr::detail::coarse_is_local(piece))
                from[target.fine.rank].push_back(&piece);
            continue;
        }
        if (!amr::detail::coarse_is_local(piece))
        {
            pack_coarsened(plan, target, access, fine_ld, op, plan.sent[piece.coarse.patch.rank]);
            continue;
        }
        bytes.clear();
        pack_coarsened(plan, target, access, fine_ld, op, bytes);
        std::byte const* at = bytes.data();
        unpack_coarsened(plan, piece, access, coarse_ld, at);
    }
    auto sends = detail::send_all(plan.sent, comm);

    detail::receive_all(from, plan.received, comm, [&](auto const& piece, auto& at) {
        unpack_coarsened(plan, piece, access, coarse_ld, at);
    });
    MPI_Waitall(static_cast<int>(sends.size()), sends.data(), MPI_STATUSES_IGNORE);
}

template<std::size_t dim, typename FineLD, typename CoarseLD>
    requires amr::LevelDataLike<CoarseLD>
void coarsen(auto op, amr::LevelMap<dim>& map, std::vector<amr::LevelPair> const& pairs,
             FineLD& fine_ld, CoarseLD& coarse_ld, MPI_Comm const comm = MPI_COMM_WORLD)
{
    coarsen(op, map, pairs, amr::by_name, fine_ld, coarse_ld, comm);
}

} // namespace damn::mpi

#endif // DAMN_MPI_INTER_LEVEL_HPP
