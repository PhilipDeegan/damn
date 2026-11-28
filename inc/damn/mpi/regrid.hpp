#ifndef DAMN_MPI_REGRID_HPP
#define DAMN_MPI_REGRID_HPP

// damn::amr regrids over MPI (damn/amr/regrid.hpp): the same regrid, with patches moving
// between ranks; collective over the ranks holding either level's patches
//
// memory: transfers to another rank are packed into that peer's buffer as each old patch is
// visited (then freed), and sent once the buffer reaches chunk bytes. While more than
// max_in_flight bytes are being sent, the rank receives what has arrived (and frees completed
// sends) before packing more. Messages hold whole transfers, in map order: MPI keeps the
// messages of a peer in order, so a receiver unpacks them against its own map

#include "damn/core/def.hpp"
#include "damn/core/logger.hpp"

#include "damn/amr/level_data.hpp"
#include "damn/amr/neighbours.hpp"
#include "damn/amr/regrid.hpp"

#include "damn/mpi/damn_mpi.hpp"

#include <cstddef>
#include <initializer_list>
#include <limits>
#include <list>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace damn::mpi
{
inline constexpr int regrid_tag = 0xdb;

struct RegridLimits
{
    std::size_t chunk         = std::size_t{64} << 20;  // send a peer's buffer at this size
    std::size_t max_in_flight = std::size_t{256} << 20; // sent, not yet received, by this rank
};

template<std::size_t dim, typename PatchData, typename Make>
MKN_KUL_FN_SIG_RET amr::LevelData<dim, PatchData>
regrid(amr::RegridMap<dim> const& map, amr::LevelData<dim, PatchData>&& old, Make&& make,
       RegridLimits const limits = {}, MPI_Comm const comm = MPI_COMM_WORLD)
{
    using namespace amr;
    DAMN_LOG_SCOPE(1, "damn::mpi::regrid");

    auto const as_int = [](std::size_t const bytes) {
        if (bytes > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("damn::mpi::regrid: a message over 2GB");
        return static_cast<int>(bytes);
    };

    // what comes from each peer, in map order
    std::map<int, std::vector<Transfer<dim> const*>> incoming;
    std::size_t expected = 0;
    for (auto const& t : map.transfers)
        if (t.src == not_local)
        {
            incoming[t.from.rank].push_back(&t);
            ++expected;
        }
    std::map<int, std::size_t> cursor; // next incoming transfer per peer

    struct Send
    {
        std::vector<std::byte> buffer;
        MPI_Request request;
    };
    std::list<Send> sends; // in flight
    std::size_t in_flight = 0;
    std::map<int, std::vector<std::byte>> buffers; // per peer, being packed

    std::vector<std::byte> received;

    auto const receive_one = [&](MPI_Status const& status, auto& new_data) {
        int bytes = 0;
        MPI_Get_count(&status, MPI_BYTE, &bytes);
        received.resize(static_cast<std::size_t>(bytes));
        MPI_Recv(received.data(), bytes, MPI_BYTE, status.MPI_SOURCE, regrid_tag, comm,
                 MPI_STATUS_IGNORE);

        auto const peer      = status.MPI_SOURCE;
        auto const& from     = incoming[peer];
        auto& next           = cursor[peer];
        std::byte const* at  = received.data();
        std::byte const* end = received.data() + received.size();
        while (at < end)
        {
            if (next == from.size())
                throw std::runtime_error("damn::mpi::regrid: rank " + std::to_string(peer)
                                         + " sent more than this rank's map expects");
            auto const& t = *from[next++];
            amr::detail::unpack_transfer(map, t, new_data(t.dst), at);
            --expected;
        }
        if (at != end)
            throw std::runtime_error("damn::mpi::regrid: rank " + std::to_string(peer)
                                     + "'s message does not match this rank's map");
    };

    // receives whatever has arrived, frees completed sends
    auto const progress = [&](auto& new_data) {
        int flag = 1;
        while (flag and expected)
        {
            MPI_Status status;
            MPI_Iprobe(MPI_ANY_SOURCE, regrid_tag, comm, &flag, &status);
            if (flag)
                receive_one(status, new_data);
        }
        for (auto it = sends.begin(); it != sends.end();)
        {
            int done = 0;
            MPI_Test(&it->request, &done, MPI_STATUS_IGNORE);
            if (done)
            {
                in_flight -= it->buffer.size();
                it = sends.erase(it);
            }
            else
                ++it;
        }
    };

    auto const flush = [&](int const peer, std::vector<std::byte>& buffer) {
        auto& s = sends.emplace_back(Send{std::move(buffer), MPI_REQUEST_NULL});
        buffer  = {};
        in_flight += s.buffer.size();
        MPI_Isend(s.buffer.data(), as_int(s.buffer.size()), MPI_BYTE, peer, regrid_tag, comm,
                  &s.request);
    };

    return amr::detail::regrid(
        map, std::move(old), make,
        [&](Transfer<dim> const& t, PatchData& src, auto& new_data) { // send
            auto& buffer = buffers[t.to.rank];
            amr::detail::pack_transfer(map, t, src, buffer);
            if (buffer.size() >= limits.chunk)
                flush(t.to.rank, buffer);
            if (in_flight > limits.max_in_flight)
            {
                DAMN_LOG_SCOPE(2, "damn::mpi::regrid::over_max_in_flight");
                while (in_flight > limits.max_in_flight)
                    progress(new_data);
            }
        },
        [&](auto& new_data) { // finish
            for (auto& [peer, buffer] : buffers)
                if (!buffer.empty())
                    flush(peer, buffer);
            DAMN_LOG_SCOPE(2, "damn::mpi::regrid::receive"); // waits for other ranks
            while (expected)
            {
                MPI_Status status;
                MPI_Probe(MPI_ANY_SOURCE, regrid_tag, comm, &status);
                receive_one(status, new_data);
            }
            for (auto& s : sends)
                MPI_Wait(&s.request, MPI_STATUS_IGNORE);
            sends.clear();
        });
}

} // namespace damn::mpi

#endif // DAMN_MPI_REGRID_HPP
