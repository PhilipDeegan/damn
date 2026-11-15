#ifndef DAMN_CORE_DEF_BOOTSTRAP_HPP
#define DAMN_CORE_DEF_BOOTSTRAP_HPP

// what this machine and build can do, from etc/bootstrap/bootstrap.py's probes (one binary per
// capability: built, run, passed): _gen_bootstrap.hpp, generated, not committed. Without it every
// capability is off. A run still checks for itself what it uses (another node, thread level)
//
//   DAMN_HAVE_BOOTSTRAP              the generated header is there
//   DAMN_HAVE_MPI_THREAD_MULTIPLE    MPI_THREAD_MULTIPLE, threads exchanging at once
//   DAMN_HAVE_MPI_CUDA_AWARE         Open MPI CUDA aware, a driver at run time
//   DAMN_HAVE_HDF5_PARALLEL          parallel HDF5 (MPI-IO), collective round trip
//   DAMN_HAVE_HDF5_SUBFILING         the subfiling VFD round trip (at MPI_THREAD_MULTIPLE)
//   DAMN_HAVE_CUDA_DEVICE            a CUDA device
//   DAMN_HAVE_MPI_VERSION            MPI builds and runs (damn::bootstrap::mpi_version_library)
//
//   damn::bootstrap::config()        every fact as a map (bootstrap.json's): "<area>_<name>" "1" /
//                                    "0", "<area>_<name>_<key>" its values, "settings"; empty
//                                    without the generated header. For python, as PHARE's
//                                    build_config

#if __has_include("damn/core/def/_gen_bootstrap.hpp")
#include "damn/core/def/_gen_bootstrap.hpp" // IWYU pragma: export
#endif

#ifndef DAMN_HAVE_BOOTSTRAP
#define DAMN_HAVE_BOOTSTRAP 0
#endif
#ifndef DAMN_HAVE_MPI_VERSION
#define DAMN_HAVE_MPI_VERSION 0
#endif
#ifndef DAMN_HAVE_MPI_THREAD_MULTIPLE
#define DAMN_HAVE_MPI_THREAD_MULTIPLE 0
#endif
#ifndef DAMN_HAVE_MPI_CUDA_AWARE
#define DAMN_HAVE_MPI_CUDA_AWARE 0
#endif
#ifndef DAMN_HAVE_HDF5_PARALLEL
#define DAMN_HAVE_HDF5_PARALLEL 0
#endif
#ifndef DAMN_HAVE_HDF5_SUBFILING
#define DAMN_HAVE_HDF5_SUBFILING 0
#endif
#ifndef DAMN_HAVE_CUDA_DEVICE
#define DAMN_HAVE_CUDA_DEVICE 0
#endif

#include <array>
#include <map>
#include <string>
#include <string_view>
#include <utility>

namespace damn::bootstrap
{
#if !DAMN_HAVE_BOOTSTRAP
inline constexpr std::array<std::pair<std::string_view, std::string_view>, 0> entries{};
#endif

inline std::map<std::string, std::string> config()
{
    std::map<std::string, std::string> out;
    for (auto const& [k, v] : entries)
        out.emplace(k, v);
    return out;
}
} // namespace damn::bootstrap

#endif // DAMN_CORE_DEF_BOOTSTRAP_HPP
