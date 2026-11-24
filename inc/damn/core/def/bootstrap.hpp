#ifndef DAMN_CORE_DEF_BOOTSTRAP_HPP
#define DAMN_CORE_DEF_BOOTSTRAP_HPP

// what this machine and build can do, from etc/bootstrap/bootstrap.py (its toolchain discovery,
// then one probe binary per capability: built, run, passed): _gen_bootstrap.hpp, generated, not
// committed; without it bootstrap_defaults.hpp (committed, every capability off). A run still
// checks for itself what it uses (another node, thread level)
//
// compile time
//   DAMN_HAVE_BOOTSTRAP, DAMN_HAVE_<AREA>_<PROBE>   1 / 0, for #if
//   damn::bootstrap::Toolchain::<flag>              discover.py's: mpi_cuda_aware, hdf5_subfiling,
//                                                   cuda_found, ... (bool); gpu ("none", "cuda",
//                                                   "rocm")
//   damn::bootstrap::<Area>::<probe>                each probe passed (Mpi::thread_multiple, ...)
//
// run time: every fact a string by path
//   damn::bootstrap::entries                        ("<area>/<probe>/<key>", value) pairs: also
//                                                   "<area>/<probe>/ok", "toolchain/<area>/<key>",
//                                                   "settings"; empty without a bootstrap
//   damn::bootstrap::get<T>(path, fallback)         a fact as T (kul's from_string), fallback
//                                                   without it: get<double>(
//                                                   "mpi/exchange_node/latency_s", 1e-6)
//   damn::bootstrap::tree()                         the facts as a bon tree (damn/core/bon.hpp):
//                                                   bon::find(tree(), "mpi") for an area's

#if __has_include("damn/core/def/_gen_bootstrap.hpp")
#include "damn/core/def/_gen_bootstrap.hpp" // IWYU pragma: export
#else
#include "damn/core/def/bootstrap_defaults.hpp" // IWYU pragma: export
#endif

#include "damn/core/bon.hpp"
#include "mkn/kul/lang/cpp/compiler/def.hpp"

#include <array>
#include <string_view>

namespace damn::bootstrap
{
// a fact as T, else fallback (no bootstrap, its probe failed or was skipped, or not a T)
template<typename T>
MKN_KUL_FN_SIG_RET T get(std::string_view const path, T const& fallback)
{
    return bon::get(entries, path, fallback);
}

// the facts as a tree of paths, made once
MKN_KUL_FN_SIG_RET inline bon::Node const& tree()
{
    static bon::Node const t = bon::tree(entries);
    return t;
}
} // namespace damn::bootstrap

#endif // DAMN_CORE_DEF_BOOTSTRAP_HPP
