#ifndef DAMN_CORE_DEF_BOOTSTRAP_HPP
#define DAMN_CORE_DEF_BOOTSTRAP_HPP

// what this machine and build can do, from etc/bootstrap/bootstrap.py: doc/core/def/bootstrap.md
// a run still checks for itself what it uses (another node, thread level)

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
