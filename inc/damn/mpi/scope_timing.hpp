#ifndef DAMN_MPI_SCOPE_TIMING_HPP
#define DAMN_MPI_SCOPE_TIMING_HPP

// phlop scope timing per MPI rank: what DAMN_LOG_SCOPE records (time, and memory or hardware
// counters per log level, see damn/core/def/phlop.hpp), one file per rank, read with phlop's
// python tools (python3 -m phlop.timing.scope_timer times|perf|variance ...)
//
// start after MPI_Init, stop before MPI_Finalize: both are collective (they synchronise the
// ranks' clocks, so phlop can compare when scopes start across ranks). Without phlop they do
// nothing

#include "damn/core/def.hpp"
#include "damn/core/def/phlop.hpp"
#include "damn/core/logger.hpp"

#include "damn/mpi/damn_mpi.hpp"

#include <cstdlib>
#include <string>

namespace damn::mpi
{
// true if the env variable is 1 or true
MKN_KUL_FN_SIG_RET inline bool env_flag(char const* const name)
{
    auto const* const v = std::getenv(name);
    return v and (std::string{v} == "1" or std::string{v} == "true");
}

MKN_KUL_FN_SIG_RET inline bool scope_timing_active()
{
#if DAMN_HAVE_PHLOP
    return phlop::mpi::scope_timer().active;
#else
    return false;
#endif
}

// collective: file_prefix + rank + ".txt" written at stop_scope_timing; the directory must
// exist by then
inline void start_scope_timing(std::string const& file_prefix)
{
#if DAMN_HAVE_PHLOP
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    phlop::mpi::scope_timer().file_name(file_prefix + std::to_string(rank) + ".txt").init();
    phlop::mpi::sync_clock();
#else
    (void)file_prefix;
#endif
}

// collective, if started: writes the file
inline void stop_scope_timing()
{
#if DAMN_HAVE_PHLOP
    if (!scope_timing_active())
        return;
    phlop::mpi::sync_clock();
    phlop::mpi::shutdown_scope_timer();
#endif
}

// writes what was recorded so far and starts again, to the same file (e.g. between
//  simulations in one process: the last one's file is kept)
inline void reset_scope_timing()
{
#if DAMN_HAVE_PHLOP
    phlop::mpi::reset_scope_timer();
#endif
}

} // namespace damn::mpi

#endif // DAMN_MPI_SCOPE_TIMING_HPP
