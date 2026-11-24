#ifndef DAMN_DAMN_CORE_HPP
#define DAMN_DAMN_CORE_HPP

// everything damn: index space, storage, tiling, execution, patch hierarchy
// damn is agnostic to what the data is; users bring its shape

#include "damn/amr/hierarchy.hpp"                      // IWYU pragma: export
#include "damn/amr/lattice.hpp"                        // IWYU pragma: export
#include "damn/amr/kelvin/sites.hpp"                   // IWYU pragma: export
#include "damn/amr/kelvin/site_data.hpp"               // IWYU pragma: export
#include "damn/amr/box_data.hpp"                       // IWYU pragma: export
#include "damn/amr/level_data.hpp"                     // IWYU pragma: export
#include "damn/amr/level_map.hpp"                      // IWYU pragma: export
#include "damn/amr/copy.hpp"                           // IWYU pragma: export
#include "damn/amr/neighbours.hpp"                     // IWYU pragma: export
#include "damn/amr/particle_data.hpp"                  // IWYU pragma: export
#include "damn/amr/topology.hpp"                       // IWYU pragma: export
#include "damn/core/data/ndarray/ndarray_base.hpp"     // IWYU pragma: export
#include "damn/core/data/ndarray/ndarray_mask.hpp"     // IWYU pragma: export
#include "damn/core/data/ndarray/ndarray_vector.hpp"   // IWYU pragma: export
#include "damn/core/data/ndarray/ndarray_view.hpp"     // IWYU pragma: export
#include "damn/core/data/tiles/tile_set.hpp"           // IWYU pragma: export
#include "damn/core/data/tiles/tile_set_mapper.hpp"    // IWYU pragma: export
#include "damn/core/data/tiles/tile_set_overlaps.hpp"  // IWYU pragma: export
#include "damn/core/data/tiles/tile_set_traversal.hpp" // IWYU pragma: export
#include "damn/core/data/tiles/tiling_options.hpp"     // IWYU pragma: export
#include "damn/core/def.hpp"                           // IWYU pragma: export
#include "damn/core/def/constants.hpp"                 // IWYU pragma: export
#include "damn/core/def/damn_config.hpp"               // IWYU pragma: export
#include "damn/core/def/phlop.hpp"                     // IWYU pragma: export
#include "damn/core/errors.hpp"                        // IWYU pragma: export
#include "damn/core/exec/kernels.hpp"                  // IWYU pragma: export
#include "damn/core/exec/kernels/launchers.hpp"        // IWYU pragma: export
#include "damn/core/exec/kernels/region_chunks.hpp"    // IWYU pragma: export
#include "damn/core/exec/execution_model.hpp"          // IWYU pragma: export
#include "damn/core/exec/leased_pool.hpp"              // IWYU pragma: export
#include "damn/core/exec/thread_pool.hpp"              // IWYU pragma: export
#include "damn/core/geometry/box.hpp"                  // IWYU pragma: export
#include "damn/core/geometry/box_spans.hpp"            // IWYU pragma: export
#include "damn/core/geometry/point.hpp"                // IWYU pragma: export
#include "damn/core/geometry/range.hpp"                // IWYU pragma: export
#include "damn/core/logger.hpp"                        // IWYU pragma: export
#include "damn/core/math/random.hpp"                   // IWYU pragma: export
#include "damn/core/math/sort.hpp"                     // IWYU pragma: export
#include "damn/core/memory/memory.hpp"                 // IWYU pragma: export
#include "damn/core/memory/span.hpp"                   // IWYU pragma: export
#include "damn/core/operators.hpp"                     // IWYU pragma: export
#include "damn/core/sys/logger_defaults.hpp"           // IWYU pragma: export
#include "damn/core/sys/monitoring.hpp"                // IWYU pragma: export
#include "damn/core/vector.hpp"                        // IWYU pragma: export

#endif /* DAMN_DAMN_CORE_HPP */
