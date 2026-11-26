#ifndef DAMN_CORE_LOGGER_HPP
#define DAMN_CORE_LOGGER_HPP

#include "damn/core/def.hpp" // IWYU pragma: keep

#include <chrono>
#include <string>
#include <cstdint>
#include <utility>
#include <memory>
#include <fstream>
#include <iostream>
#include <ctime>

#if !defined(DAMN_LOG_LEVEL)
#define DAMN_LOG_LEVEL 2 // 0 == off
#endif

namespace damn
{
constexpr static std::uint8_t LOG_LEVEL = DAMN_LOG_LEVEL;
} // namespace damn

#if !defined(NDEBUG) || defined(DAMN_FORCE_DEBUG_DO) || defined(DAMN_FORCE_LOG_LINE)
#include <sstream>  // IWYU pragma: keep
#include <iostream> // IWYU pragma: keep

#define DAMN_LOG_LINE_STR(str)                                                                    \
    std::cout << __FILE__ << ":" << __LINE__ << " - " << (str) << std::endl
// named stream: an rvalue stream would resolve `<< s` from inside std, by ADL only
#define DAMN_LOG_LINE_SS(s)                                                                       \
    DAMN_LOG_LINE_STR(([&] {                                                                      \
        std::stringstream damn_log_ss_;                                                           \
        damn_log_ss_ << s;                                                                        \
        return damn_log_ss_.str();                                                                \
    }()))
#else
#define DAMN_LOG_LINE_STR(str)
#define DAMN_LOG_LINE_SS(str)
#endif
#define DAMN_LOG_LINE DAMN_LOG_LINE_STR("")


#if DAMN_WITH_CALIPER
#include "caliper/cali.h" // IWYU pragma: keep


#define DAMN_LOG_START(lvl, str) CALI_MARK_BEGIN(str)
#define DAMN_LOG_STOP(lvl, str) CALI_MARK_END(str)
#define DAMN_LOG_SCOPE(lvl, str) damn::scope_log DAMN_STR_CAT(__damn_scope, __LINE__)(lvl, str)

#else // !DAMN_WITH_CALIPER

#include "damn/core/sys/logger_defaults.hpp" // IWYU pragma: keep


#endif // DAMN_WITH_CALIPER

namespace damn
{
// std::cout to log_out for this object's lifetime. It owns the stream, so the redirected buffer
// never outlives the object that installed it. Hold it as a member, not as a destructor body's
// job: a throwing constructor runs only its built members' destructors, not its own
class CoutRedirect
{
public:
    explicit CoutRedirect(std::unique_ptr<std::ofstream> log_out)
        : log_out_{std::move(log_out)}
    {
        if (log_out_)
            saved_ = std::cout.rdbuf(log_out_->rdbuf());
    }

    ~CoutRedirect()
    {
        if (saved_)
            std::cout.rdbuf(saved_);
    }

    CoutRedirect(CoutRedirect const&)            = delete;
    CoutRedirect& operator=(CoutRedirect const&) = delete;

private:
    std::unique_ptr<std::ofstream> log_out_;
    std::streambuf* saved_ = nullptr;
};




struct scope_log
{
    scope_log(int&& i_, std::string&& str)
        : i{i_}
        , key{std::move(str)}
    {
        if (i <= LOG_LEVEL)
        {
            DAMN_LOG_START(i, key.c_str());
        }
    }
    ~scope_log()
    {
        if (i <= LOG_LEVEL)
        {
            DAMN_LOG_STOP(i, key.c_str());
        }
    }

    int i;
    std::string key;
};


#if !defined(NDEBUG) || defined(DAMN_FORCE_DEBUG_DO) || defined(DAMN_FORCE_LOG_LINE)
struct ScopeTimer
{
    std::string key;
    std::size_t start_time = now();

    std::string static inline const FORMAT = ":%Y-%m-%d-%H:%M:%S";

    ~ScopeTimer()
    {
        std::cout << formated_time() << " DAMN SCOPE TIMER: " << key
                  << " time: " << (now() - start_time) / 1e6 << " ms" << std::endl;
    }

    std::uint64_t static now()
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    std::string static formated_time()
    {
        char date[256];
        auto now       = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);

        std::strftime(date, sizeof(date), FORMAT.c_str(), std::gmtime(&in_time_t));
        return date;
    }
};
#endif //


} // namespace damn

#if !defined(NDEBUG) || defined(DAMN_FORCE_DEBUG_DO) || defined(DAMN_FORCE_LOG_LINE)
#define DAMN_FN_TIMER(key) damn::ScopeTimer DAMN_STR_CAT(__damn_scope_, __LINE__)(key)
#else
#define DAMN_FN_TIMER(key) // noop
#endif                      //

#endif /* DAMN_CORE_LOGGER_H */
