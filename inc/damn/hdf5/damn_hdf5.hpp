#ifndef DAMN_HDF5_DAMN_HDF5_HPP
#define DAMN_HDF5_DAMN_HDF5_HPP

// include damn's HDF5 through this: DAMN_HAS_HIGHFIVE is 1 when HighFive is on the include
// path (the damn_hdf5 profile fetches it), and DAMN_WITH_HIGHFIVE(...) keeps code that
// needs it out of builds without it

#if __has_include("highfive/H5File.hpp") and (!defined(DAMN_HAS_HIGHFIVE) || DAMN_HAS_HIGHFIVE == 1)

#include "damn/hdf5/h5_file.hpp" // IWYU pragma: export

#include "highfive/H5Version.hpp"

#ifndef HIGHFIVE_VERSION_STRING
#pragma message("Highfive must be at least version 2.7.1")
#error // highfive is too old
#endif

#undef DAMN_HAS_HIGHFIVE
#define DAMN_HAS_HIGHFIVE 1
#define DAMN_WITH_HIGHFIVE(...) __VA_ARGS__

#else

#undef DAMN_HAS_HIGHFIVE
#define DAMN_HAS_HIGHFIVE 0
#define DAMN_WITH_HIGHFIVE(...)

#endif

#endif // DAMN_HDF5_DAMN_HDF5_HPP
