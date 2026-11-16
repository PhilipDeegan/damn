#ifndef DAMN_CORE_DEF_THRUST_HPP
#define DAMN_CORE_DEF_THRUST_HPP




// thrust whenever its headers are found (DAMN_HAVE_THRUST=0 to opt out): the GPU's (CUDA's CCCL,
//  ROCm's rocThrust), or on the host CCCL's (damn's base profile clones it, tpp/cccl) with its CPP
//  device system, serial on the host: thrust::device and device / universal vectors are host
//  memory there. THRUST_DEVICE_SYSTEM must be the same in every translation unit: damn's headers
//  first, or -DTHRUST_DEVICE_SYSTEM=... on the command line

// clang-format off
#if !defined(DAMN_HAVE_THRUST) && __has_include(<thrust/iterator/zip_iterator.h>)
  #define DAMN_HAVE_THRUST 1
#endif // DAMN_HAVE_THRUST

#if !defined(DAMN_HAVE_THRUST)
  #define DAMN_HAVE_THRUST 0
#endif

#if DAMN_HAVE_THRUST && !DAMN_HAVE_GPU && !defined(THRUST_DEVICE_SYSTEM)
  #define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CPP
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
