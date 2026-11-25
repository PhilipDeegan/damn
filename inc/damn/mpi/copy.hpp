#ifndef DAMN_MPI_COPY_HPP
#define DAMN_MPI_COPY_HPP

// damn::amr copies between patches of any rank, collective: doc/mpi/copy.md
// every source is read (1, 2) before anything is written (3, 4), so sums and maxes use the
// values from before the copy. Device resources take a device path: doc/mpi/copy.md
// damn::mpi::Copy spreads the steps over the level's work: doc/mpi/copy.md#in-flight

#include "damn/core/logger.hpp"
#include "damn/amr/copy.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/mpi/damn_mpi.hpp"
#include "damn/core/memory/scratch.hpp"
#include "damn/core/def/damn_config.hpp"
#include "mkn/kul/lang/cpp/compiler/def.hpp"

#if DAMN_HAVE_GPU && __has_include(<mpi-ext.h>)
#include <mpi-ext.h> // MPIX_Query_cuda_support, MPIX_Query_rocm_support (Open MPI)
#endif

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
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

    inline bool copy_in_flight = false; // one damn::mpi::Copy per rank at a time: one tag (#57)

    // damn::mpi::Copy's schedule, built once per plan (plan.in_flight), with its per copy state
    struct InFlight
    {
        using Clock = std::chrono::steady_clock;

        std::vector<int> ranks{};             // per peer, rank order
        std::vector<std::size_t> pieces_at{}; // per peer + 1, into pieces (CSR)
        std::vector<std::size_t> pieces{};    // into plan.outgoing, plan order: the message's
        std::vector<std::size_t> feeds_at{};  // per local patch + 1, into feeds (CSR)
        std::vector<std::size_t> feeds{};     // peers, a patch feeding one counted once
        std::vector<std::size_t> feeders{};   // per peer: local patches it waits for
        std::vector<std::size_t> remote_first{};

        std::vector<std::size_t> waiting{};
        std::vector<std::uint8_t> sent{}, done{}, received{}; // per peer, patch, plan.from entry
        std::vector<MPI_Request> sends{};
        std::vector<MPI_Status> probed{}; // simulate_latency's, per plan.from entry
        std::vector<std::optional<Clock::time_point>> due{};
    };

    template<std::size_t dim>
    InFlight& in_flight(amr::Plan<dim>& plan, std::size_t const n_local)
    {
        if (plan.in_flight)
            return *static_cast<InFlight*>(plan.in_flight.get());

        auto made = std::make_shared<InFlight>();
        auto& f   = *made;
        for (auto const& piece : plan.outgoing)
            f.ranks.push_back(piece.dst_rank);
        std::sort(f.ranks.begin(), f.ranks.end());
        f.ranks.erase(std::unique(f.ranks.begin(), f.ranks.end()), f.ranks.end());
        auto const peer_of = [&](int const rank) {
            return static_cast<std::size_t>(std::lower_bound(f.ranks.begin(), f.ranks.end(), rank)
                                            - f.ranks.begin());
        };

        f.pieces_at.assign(f.ranks.size() + 1, 0);
        for (auto const& piece : plan.outgoing)
            ++f.pieces_at[peer_of(piece.dst_rank) + 1];
        std::partial_sum(f.pieces_at.begin(), f.pieces_at.end(), f.pieces_at.begin());
        f.pieces.resize(plan.outgoing.size());
        {
            auto at = f.pieces_at;
            for (std::size_t i = 0; i < plan.outgoing.size(); ++i)
                f.pieces[at[peer_of(plan.outgoing[i].dst_rank)]++] = i;
        }

        std::vector<std::pair<std::size_t, std::size_t>> fed; // (local patch, peer), unique
        for (auto const& piece : plan.outgoing)
        {
            if (piece.src.local >= n_local)
                throw std::runtime_error("damn::mpi::Copy: an outgoing piece from no local patch");
            fed.emplace_back(piece.src.local, peer_of(piece.dst_rank));
        }
        std::sort(fed.begin(), fed.end());
        fed.erase(std::unique(fed.begin(), fed.end()), fed.end());
        f.feeds_at.assign(n_local + 1, 0);
        f.feeders.assign(f.ranks.size(), 0);
        for (auto const& [patch, peer] : fed)
        {
            ++f.feeds_at[patch + 1];
            ++f.feeders[peer];
            f.feeds.push_back(peer);
        }
        std::partial_sum(f.feeds_at.begin(), f.feeds_at.end(), f.feeds_at.begin());

        for (std::size_t i = 0; i < n_local; ++i)
            if (f.feeds_at[i] != f.feeds_at[i + 1])
                f.remote_first.push_back(i);
        for (std::size_t i = 0; i < n_local; ++i)
            if (f.feeds_at[i] == f.feeds_at[i + 1])
                f.remote_first.push_back(i);

        plan.in_flight = made;
        return f;
    }
} // namespace detail

// a copy in flight while the level's work runs: doc/mpi/copy.md#in-flight
// n, pairs, access and ld outlive the Copy; MPI calls from one thread (MPI_THREAD_FUNNELED)
template<amr::Region region, amr::Arrival arrival, std::size_t dim, typename Access,
         typename PatchData>
class Copy
{
    using Clock = detail::InFlight::Clock;

public:
    Copy(amr::CopyOp<region, arrival> const op, amr::Neighbours<dim>& n,
         std::vector<amr::ResourcePair> const& pairs, Access const& access,
         amr::LevelData<dim, PatchData>& ld, MPI_Comm const comm = MPI_COMM_WORLD)
        : op_{op}
        , access_{access}
        , ld_{ld}
        , comm_{comm}
        , plan_{(amr::check_access(access, pairs.size()), amr::plan_for(region, n, ld, pairs))}
        , f_{detail::in_flight(plan_, ld.patches.size())}
    {
        // before the plan's per copy state is reset: another Copy may be on the same plan
        if (detail::copy_in_flight)
            throw std::runtime_error("damn::mpi::Copy: another copy is in flight");
        detail::copy_in_flight = true;

#if DAMN_HAVE_GPU
        if constexpr (region != amr::Region::Leaving)
            on_device_ = amr::detail::device_table(plan_, access, ld).usable;
#endif
#if DAMN_HAVE_GPU && DAMN_HAVE_THRUST
        if constexpr (region == amr::Region::Leaving)
            on_device_ = amr::detail::item_table(plan_, access, ld).usable;
#endif

        f_.waiting = f_.feeders;
        f_.sent.assign(f_.ranks.size(), 0);
        f_.done.assign(ld.patches.size(), 0);
        f_.received.assign(plan_.from.size(), 0);
        f_.sends.assign(f_.ranks.size(), MPI_REQUEST_NULL);
    }

    Copy(Copy const&)            = delete;
    Copy& operator=(Copy const&) = delete;

    ~Copy() { detail::copy_in_flight = false; }

    // a dispatch order for the work before the copy (ThreadPool::PatchHooks::order)
    MKN_KUL_FN_SIG_RET std::span<std::size_t const> remote_first() const { return f_.remote_first; }

    // local patch i's sources are final
    void done(std::size_t const i)
    {
        if (on_device_ or i >= f_.done.size() or f_.done[i])
            return;
        f_.done[i] = 1;
        for (auto k = f_.feeds_at[i]; k < f_.feeds_at[i + 1]; ++k)
            if (--f_.waiting[f_.feeds[k]] == 0)
                send_(f_.feeds[k]);
    }

    // benchmarks on one node, where messages arrive at once: doc/mpi/copy.md#in-flight
    void simulate_latency(std::function<double(int, std::size_t)> latency)
    {
        latency_ = std::move(latency);
        f_.due.assign(plan_.from.size(), std::nullopt);
        f_.probed.resize(plan_.from.size());
    }

    // receives what has arrived, writes nothing
    void poll()
    {
        if (on_device_)
            return;
        for (std::size_t j = 0; j < plan_.from.size(); ++j)
        {
            if (f_.received[j])
                continue;
            if (!latency_)
            {
                int arrived = 0;
                MPI_Status status;
                MPI_Iprobe(plan_.from[j].first, copy_tag, comm_, &arrived, &status);
                if (arrived)
                    receive_(j, status);
                continue;
            }
            if (!f_.due[j])
            {
                int arrived = 0;
                MPI_Iprobe(plan_.from[j].first, copy_tag, comm_, &arrived, &f_.probed[j]);
                if (arrived)
                    f_.due[j] = due_of_(j);
            }
            if (f_.due[j] and Clock::now() >= *f_.due[j])
                receive_(j, f_.probed[j]);
        }
    }

    void finish()
    {
        using namespace amr;
        DAMN_LOG_SCOPE(2, "damn::mpi::copy<" + std::string{name(op_)} + ">");

#if DAMN_HAVE_GPU
        if constexpr (region != Region::Leaving)
            if (on_device_)
                return detail::copy_on_device<arrival>(
                    plan_, amr::detail::device_table(plan_, access_, ld_), ld_, comm_);
#endif
#if DAMN_HAVE_GPU && DAMN_HAVE_THRUST
        if constexpr (region == Region::Leaving)
            if (on_device_)
                return detail::items_on_device(plan_, amr::detail::item_table(plan_, access_, ld_),
                                               access_, ld_, comm_);
#endif

        // 1. send
        {
            DAMN_LOG_SCOPE(2, "damn::mpi::copy::pack_send");
            for (std::size_t p = 0; p < f_.ranks.size(); ++p)
                send_(p);
        }

        // 2. local
        {
            DAMN_LOG_SCOPE(2, "damn::mpi::copy::local");
            if constexpr (arrival == Arrival::Assign)
                amr::detail::assign(plan_, access_, ld_);
            else if constexpr (arrival == Arrival::Append)
                amr::detail::append_items(plan_, access_, ld_);
            else
                amr::detail::pack_local(plan_, access_, ld_);
        }

        // 3. receive: pieces from each peer, in plan order
        if constexpr (arrival == Arrival::Sum)
            amr::detail::local_sources(plan_); // and the received ones below: plan.sources
        for (std::size_t j = 0; j < plan_.from.size(); ++j)
        {
            auto const& [peer, indices] = plan_.from[j];
            if (!f_.received[j])
                receive_(j, wait_(j));

            DAMN_LOG_SCOPE(2, "damn::mpi::copy::unpack");
            auto const& buffer  = plan_.received[peer];
            std::byte const* at = buffer.data();
            for (auto const i : indices)
            {
                DAMN_LOG_SCOPE(3, "damn::mpi::copy::unpack::piece"); // one patch overlap
                auto const& piece = plan_.pieces[i];
                if constexpr (arrival == Arrival::Sum)
                {
                    if (at + piece.bytes > buffer.data() + buffer.size())
                        break; // checked below
                    plan_.sources[i] = at;
                    at += piece.bytes;
                }
                else
                    unpack_piece<arrival>(region, piece, access_, ld_, at);
            }
            if (at != buffer.data() + buffer.size())
                throw std::runtime_error("damn::mpi::copy: rank " + std::to_string(peer)
                                         + "'s message does not match this rank's plan");
        }

        // 4. finish
        {
            DAMN_LOG_SCOPE(2, "damn::mpi::copy::finish");
            if constexpr (arrival == Arrival::Sum)
                amr::detail::fold_sums(plan_, access_, ld_);
            else if constexpr (arrival == Arrival::Max)
                amr::detail::unpack_local<arrival>(plan_, access_, ld_);
            else if constexpr (arrival == Arrival::Append) // local and received items in, each once
                amr::detail::appended(plan_, access_, ld_);
        }
        DAMN_LOG_SCOPE(2, "damn::mpi::copy::sends"); // until the peers have received them
        MPI_Waitall(static_cast<int>(f_.sends.size()), f_.sends.data(), MPI_STATUSES_IGNORE);
    }

private:
    void send_(std::size_t const p)
    {
        if (f_.sent[p])
            return;
        auto& buffer = plan_.sent[f_.ranks[p]];
        buffer.clear();
        for (auto k = f_.pieces_at[p]; k < f_.pieces_at[p + 1]; ++k)
        {
            DAMN_LOG_SCOPE(3, "damn::mpi::copy::pack_send::piece"); // one patch overlap
            amr::pack_piece(region, plan_.outgoing[f_.pieces[k]], access_, ld_, buffer);
        }
        MPI_Isend(buffer.data(), detail::as_count(buffer.size()), MPI_BYTE, f_.ranks[p], copy_tag,
                  comm_, &f_.sends[p]);
        f_.sent[p] = 1;
    }

    MKN_KUL_FN_SIG_RET MPI_Status wait_(std::size_t const j)
    {
        DAMN_LOG_SCOPE(2, "damn::mpi::copy::wait"); // for the peer's message
        if (!latency_)
        {
            MPI_Status status;
            MPI_Probe(plan_.from[j].first, copy_tag, comm_, &status);
            return status;
        }
        if (!f_.due[j])
        {
            MPI_Probe(plan_.from[j].first, copy_tag, comm_, &f_.probed[j]);
            f_.due[j] = due_of_(j);
        }
        std::this_thread::sleep_until(*f_.due[j]);
        return f_.probed[j];
    }

    MKN_KUL_FN_SIG_RET Clock::time_point due_of_(std::size_t const j) const
    {
        int bytes = 0;
        MPI_Get_count(&f_.probed[j], MPI_BYTE, &bytes);
        auto const s = latency_(plan_.from[j].first, static_cast<std::size_t>(bytes));
        return Clock::now()
               + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(s));
    }

    void receive_(std::size_t const j, MPI_Status const& status)
    {
        auto const peer = plan_.from[j].first;
        int bytes       = 0;
        MPI_Get_count(&status, MPI_BYTE, &bytes);
        auto& buffer = plan_.received[peer];
        buffer.resize(static_cast<std::size_t>(bytes));
        DAMN_LOG_SCOPE(2, "damn::mpi::copy::recv"); // the probed message out of MPI
        MPI_Recv(buffer.data(), bytes, MPI_BYTE, peer, copy_tag, comm_, MPI_STATUS_IGNORE);
        f_.received[j] = 1;
    }

    amr::CopyOp<region, arrival> op_;
    Access const& access_;
    amr::LevelData<dim, PatchData>& ld_;
    MPI_Comm comm_;
    amr::Plan<dim>& plan_;
    detail::InFlight& f_;
    bool on_device_ = false; // one launch packs every piece: done() and poll() wait for finish()
    std::function<double(int, std::size_t)> latency_{};
};

// access: amr::by_name, or accessors one per pair
template<amr::Region region, amr::Arrival arrival, std::size_t dim, typename Access,
         typename PatchData>
void copy(amr::CopyOp<region, arrival> const op, amr::Neighbours<dim>& n,
          std::vector<amr::ResourcePair> const& pairs, Access const& access,
          amr::LevelData<dim, PatchData>& ld, MPI_Comm const comm = MPI_COMM_WORLD)
{
    Copy{op, n, pairs, access, ld, comm}.finish();
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
