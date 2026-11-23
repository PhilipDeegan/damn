#ifndef DAMN_CORE_DEF_HPP
#define DAMN_CORE_DEF_HPP

#include "damn/core/def/damn_config.hpp" // IWYU pragma: keep

#include "mkn/kul/lang/cpp/compiler/def/inline.hpp" // IWYU pragma: export

#include <cassert>
#include <type_traits>

#define NO_DISCARD [[nodiscard]]

// inline regardless of the compiler's heuristics: mkn.kul's MKN_KUL_ALWAYS_INLINE (a function),
//  MKN_KUL_LAMBDA_ALWAYS_INLINE (after a lambda's parameter list), MKN_KUL_INLINE_CALL (a call).
//  The DAMN_ names are aliases until their users move to kul's
#define DAMN_LAMBDA_ALWAYS_INLINE MKN_KUL_LAMBDA_ALWAYS_INLINE
#define DAMN_ALWAYS_INLINE MKN_KUL_ALWAYS_INLINE

#if !defined(NDEBUG) || defined(DAMN_FORCE_DEBUG_DO)
#define DAMN_DEBUG_DO(...) __VA_ARGS__
#define DAMN_DEBUG 1
#else
#define DAMN_DEBUG_DO(...)
#define DAMN_DEBUG 0
#endif

// heavy checks (whole level sums, per stage invariants, ...): opt in at compile time with
//  -DDAMN_SUPER_DEBUG (1; =0 off), in any build (-g 0 too); nothing otherwise. DAMN_DEBUG_DO
//  is the light kind, on unless NDEBUG
#if !defined(DAMN_SUPER_DEBUG)
#define DAMN_SUPER_DEBUG 0
#endif
#if DAMN_SUPER_DEBUG
#define DAMN_SUPER_DEBUG_DO(...) __VA_ARGS__
#else
#define DAMN_SUPER_DEBUG_DO(...)
#endif


#if !defined(DAMN_UNDEF_ASSERT)
//  Cuda can fail to compile with assertions
//  I've seen a github issue, will ref
#define DAMN_ASSERT(...) assert(__VA_ARGS__)
#else
#define DAMN_ASSERT(...)
#endif


#define DAMN_TO_STR_IMPL(x) #x // convert macro text to string
#define DAMN_TO_STR(x) DAMN_TO_STR_IMPL(x)
#define DAMN_TOKEN_PASTE(x, y) x##y
#define DAMN_STR_CAT(x, y) DAMN_TOKEN_PASTE(x, y)


namespace damn::core
{

NO_DISCARD bool isUsable(auto const&... args)
{
    static_assert(sizeof...(args) > 0);
    auto constexpr check = [](auto const& arg) {
        if constexpr (std::is_pointer_v<std::decay_t<decltype(arg)>>)
            return arg != nullptr;
        else
            return arg.isUsable();
    };
    return (check(args) && ...);
}


NO_DISCARD bool isSettable(auto const&... args)
{
    static_assert(sizeof...(args) > 0);
    auto constexpr check = [](auto const& arg) {
        if constexpr (std::is_pointer_v<std::decay_t<decltype(arg)>>)
            return arg == nullptr;
        else
            return arg.isSettable();
    };
    return (check(args) && ...);
}

} // namespace damn::core




#endif // DAMN_CORE_DEF_HPP
