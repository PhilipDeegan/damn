#ifndef DAMN_AMR_BOX_DATA_HPP
#define DAMN_AMR_BOX_DATA_HPP

// BoxData: a resource whose values sit on the points of a box in the level (AMR) index
// space, like a field. damn moves its values (ghost fills, border sums...) through the
// user's free functions, found by ADL:
//
//   Box<int, dim> domain_box(resource, patch_box)  // the points the patch owns
//   Box<int, dim> ghost_box(resource, patch_box)   // all its points, ghosts included
//   value_type&   value(resource, patch_box, amr_point)
//
// patch_box is the patch's cell box. The boxes must depend only on the patch box (and the
// resource's kind, e.g. its centering), not on the instance's values: damn calls them with
// one patch's resource for the boxes of other patches

#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace damn::amr
{
template<typename R, std::size_t dim>
concept BoxData = requires(R& r, R const& cr, core::Box<int, dim> const& box,
                           core::Point<int, dim> const& p) {
    { domain_box(cr, box) } -> std::same_as<core::Box<int, dim>>;
    { ghost_box(cr, box) } -> std::same_as<core::Box<int, dim>>;
    value(r, box, p);
    value(cr, box, p);
};

template<typename R, std::size_t dim>
using box_data_value_t = std::decay_t<decltype(value(std::declval<R&>(),
                                                     std::declval<core::Box<int, dim>>(),
                                                     std::declval<core::Point<int, dim>>()))>;

} // namespace damn::amr

#endif // DAMN_AMR_BOX_DATA_HPP
