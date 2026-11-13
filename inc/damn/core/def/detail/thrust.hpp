#ifndef DAMN_CORE_DEF_THRUST_HPP
#define DAMN_CORE_DEF_THRUST_HPP




// clang-format off
#if DAMN_HAVE_GPU && !defined(DAMN_HAVE_THRUST) && __has_include(<thrust/iterator/zip_iterator.h>)
  #define DAMN_HAVE_THRUST 1
#endif // DAMN_HAVE_THRUST

#if !defined(DAMN_HAVE_THRUST)
  #define DAMN_HAVE_THRUST 0
#endif


#if DAMN_HAVE_THRUST

  #include <thrust/sort.h>
  #include <thrust/execution_policy.h>
  #include <thrust/iterator/zip_iterator.h>

  #define DAMN_WITH_THRUST(...) __VA_ARGS__
  #define DAMN_WITH_THRUST_ELSE(...)
  #define DAMN_WITH_THRUST_ELSE_THROW(...) __VA_ARGS__

#else // !__has_include(<thrust/iterator/zip_iterator.h>)

  #define DAMN_HAVE_THRUST 0
  #define DAMN_WITH_THRUST(...)
  #define DAMN_WITH_THRUST_ELSE(...) __VA_ARGS__
  #define DAMN_WITH_THRUST_ELSE_THROW(...) throw std::runtime_error("Thrust not found!");

#endif // DAMN_HAVE_THRUST

// clang-format on

#endif /* DAMN_CORE_DEF_THRUST_HPP */
