#ifndef DAMN_CORE_DEF_HPP
#define DAMN_CORE_DEF_HPP

#include "damn/core/def/damn_config.hpp" // IWYU pragma: keep

#include "mkn/kul/lang/cpp/compiler/def.hpp"        // IWYU pragma: export
#include "mkn/kul/lang/cpp/compiler/def/inline.hpp" // IWYU pragma: export

#include <cassert>
#include <type_traits>
#include <utility> // IWYU pragma: keep (DAMN_DAO_FN: std::forward)

// [[nodiscard]]: mkn.kul's MKN_KUL_FN_SIG_RET (no damn macro)

// inline regardless of the compiler's heuristics: mkn.kul's MKN_KUL_ALWAYS_INLINE (a function),
//  MKN_KUL_LAMBDA_ALWAYS_INLINE (after a lambda's parameter list), MKN_KUL_INLINE_CALL (a call).
//  The DAMN_ names are aliases until their users move to kul's
#define DAMN_LAMBDA_ALWAYS_INLINE MKN_KUL_LAMBDA_ALWAYS_INLINE
#define DAMN_ALWAYS_INLINE MKN_KUL_ALWAYS_INLINE

// fn(Dao const&) also callable as fn(args...): the overload builds Dao{args...} (Dao's members
//  in order, missing trailing ones defaulted) and calls fn with it, so call sites pass plain
//  arguments and need not know the DAO. Dao's template arguments come from aggregate deduction
//  (C++20, no deduction guides): each must appear in some member's type. Not used when the first
//  argument already is a Dao (Dao{dao} is then a copy of the same type). A DAO used on device
//  holds views, not references. As mkn.kul's MKN_KUL_FN_SIG / _RET:
//   DAMN_DAO_FN(spec, fn, Dao)      the overload's specifiers spec (e.g. inline, DAMN_ALL)
//   DAMN_DAO_RET_FN(spec, fn, Dao)  [[nodiscard]] too: fn's result is the point of calling it
//                                   (fn returning void is refused at compile time)
// Dao is a template name deduced from the braces: it cannot be parenthesised
// NOLINTBEGIN(bugprone-macro-parentheses)
#define DAMN_DAO_FN_IMPL_(attr, spec, fn, Dao, check)                                              \
    attr spec auto fn(auto&& first, auto&&... args)                                                \
        requires(not std::is_same_v<std::decay_t<decltype(first)>,                                 \
                                    decltype(Dao{std::forward<decltype(first)>(first),             \
                                                 std::forward<decltype(args)>(args)...})>)         \
    {                                                                                              \
        using Result_t [[maybe_unused]] = decltype(fn(                                             \
            Dao{std::forward<decltype(first)>(first), std::forward<decltype(args)>(args)...}));    \
        static_assert(check, #fn ": DAMN_DAO_RET_FN for a function returning void, use "           \
                                 "DAMN_DAO_FN");                                                   \
        return fn(                                                                                 \
            Dao{std::forward<decltype(first)>(first), std::forward<decltype(args)>(args)...});     \
    }
#define DAMN_DAO_FN(spec, fn, Dao) DAMN_DAO_FN_IMPL_(, spec, fn, Dao, true)
#define DAMN_DAO_RET_FN(spec, fn, Dao)                                                             \
    DAMN_DAO_FN_IMPL_([[nodiscard]], spec, fn, Dao, not std::is_void_v<Result_t>)
// NOLINTEND(bugprone-macro-parentheses)

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

MKN_KUL_FN_SIG_RET bool isUsable(auto const&... args)
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


MKN_KUL_FN_SIG_RET bool isSettable(auto const&... args)
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
