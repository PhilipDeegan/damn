#ifndef DAMN_CORE_OPERATORS_HPP
#define DAMN_CORE_OPERATORS_HPP

#include "damn/core/def.hpp"
#include "damn/core/meta/type_meta.hpp"
#include "damn/core/def/damn_config.hpp"
#include "mkn/gpu.hpp"

#ifndef DAMN_HAVE_GPU
#define DAMN_HAVE_GPU 0
#endif

#include <atomic>
#include <type_traits>

// device atomics (atomicAdd...) only exist in the device compilation pass (DAMN_DEVICE_PASS);
// host calls of DAMN_ALL code (e.g. GPU_UNIFIED CPU fallbacks) use the std::atomic paths


namespace damn::core
{
template<typename T, bool atomic, bool GPU>
struct Operators
{
    T static constexpr ONE = 1;

    static constexpr bool device_atomic = GPU and atomic and DAMN_DEVICE_PASS;

    static_assert(not std::is_const_v<T>); // doesn't make sense

    void operator+=(T const& v) DAMN_ALL
    {
        if constexpr (device_atomic)
        {
            atomicAdd(&t, v);
        }
        else if constexpr (atomic)
        {
            std::atomic_ref<T>{t}.fetch_add(v);
        }
        else
            t += v;
    }
    void operator+=(T const&& v) DAMN_ALL { (*this) += v; }

    void operator-=(T const& v) DAMN_ALL
    {
        if constexpr (device_atomic)
        {
            atomicSub(&t, v);
        }
        else if constexpr (atomic)
        {
            std::atomic_ref<T>{t}.fetch_sub(v);
        }
        else
            t -= v;
    }
    void operator-=(T const&& v) DAMN_ALL { (*this) -= v; }

    auto increment_return_old() DAMN_ALL // postfix increment
    {
        if constexpr (device_atomic)
        {
            auto o = atomicAdd(&t, ONE);
            DAMN_ASSERT(o < t);
            return o;
        }
        else if constexpr (atomic)
        {
            return std::atomic_ref<T>{t}.fetch_add(ONE);
        }
        else
        {
            T tmp = t;
            ++t;
            return tmp;
        }
    }

    auto add_return_old(T const v) DAMN_ALL // fetch_add
    {
        if constexpr (device_atomic)
            return atomicAdd(&t, v);
        else if constexpr (atomic)
            return std::atomic_ref<T>{t}.fetch_add(v);
        else
        {
            T tmp = t;
            t += v;
            return tmp;
        }
    }

    auto max_return_old(T const v) DAMN_ALL // atomic max, integral T
    {
        if constexpr (device_atomic)
            return atomicMax(&t, v);
        else if constexpr (atomic)
        {
            std::atomic_ref<T> ref{t};
            T old = ref.load();
            while (old < v and !ref.compare_exchange_weak(old, v))
            {
            }
            return old;
        }
        else
        {
            T tmp = t;
            t     = t < v ? v : t;
            return tmp;
        }
    }

    auto static compare_and_swap(T* addr, T compare, T value) DAMN_ALL
    {
        if constexpr (device_atomic)
        {
            return atomicCAS(addr, compare, value);
        }
        else if constexpr (atomic)
        {
            // like atomicCAS: returns the value at addr before the call, == compare on success
            std::atomic_ref<T>{*addr}.compare_exchange_strong(compare, value);
            return compare;
        }
        else
            static_assert(dependent_false_v<T>, "compare_and_swap requires atomic");
    }

    T& t;
};
} // namespace damn::core

#endif /* DAMN_CORE_OPERATORS_HPP */
