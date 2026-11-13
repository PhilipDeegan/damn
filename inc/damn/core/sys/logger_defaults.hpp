// IWYU pragma: private, include "core/logger.hpp"

#ifndef DAMN_CORE_SYS_LOGGER_DEFAULTS_HPP
#define DAMN_CORE_SYS_LOGGER_DEFAULTS_HPP

#include "damn/core/def/phlop.hpp"

// DAMN_SCOPE_TIMER / DAMN_SCOPE come from core/def/phlop.hpp when phlop is available
#ifdef DAMN_LOG_SCOPE_PRINT // print scopes instead of timing them, overrides phlop
#undef DAMN_SCOPE_TIMER
#undef DAMN_SCOPE
#define DAMN_SCOPE_TIMER(str) DAMN_LOG_LINE_SS(str) //
#endif // DAMN_LOG_SCOPE_PRINT

#ifndef DAMN_SCOPE_TIMER // uncomment next line to activate
#define DAMN_SCOPE_TIMER(str) DAMN_LOG_LINE_STR(str);
#endif // DAMN_SCOPE_TIMER

#ifndef DAMN_SCOPE // no phlop, scope flags ignored
#define DAMN_SCOPE(str, ...) DAMN_SCOPE_TIMER(str)
#endif // DAMN_SCOPE


#if DAMN_LOG_LEVEL >= 1
#define DAMN_LOG_SCOPE_1(str) DAMN_SCOPE(str, DAMN_LOG_SCOPE_FLAGS_1)
#else
#define DAMN_LOG_SCOPE_1(str)
#endif // LOG_LEVEL >= 1


#if DAMN_LOG_LEVEL >= 2
#define DAMN_LOG_SCOPE_2(str) DAMN_SCOPE(str, DAMN_LOG_SCOPE_FLAGS_2)
#else
#define DAMN_LOG_SCOPE_2(str)
#endif // LOG_LEVEL >= 2


#if DAMN_LOG_LEVEL == 3
#define DAMN_LOG_SCOPE_3(str) DAMN_SCOPE(str, DAMN_LOG_SCOPE_FLAGS_3)
#else
#define DAMN_LOG_SCOPE_3(str)
#endif // LOG_LEVEL == 3


#define DAMN_LOG_START(lvl, str)
#define DAMN_LOG_STOP(lvl, str)
#define DAMN_LOG_SCOPE(lvl, str) DAMN_STR_CAT(DAMN_LOG_SCOPE_, lvl)(str)


#endif /* DAMN_CORE_SYS_LOGGER_DEFAULTS_HPP */
