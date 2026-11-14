#ifndef DAMN_AMR_PARTICLE_DATA_HPP
#define DAMN_AMR_PARTICLE_DATA_HPP

// ParticleData: a resource holding items (particles) that each sit in a cell of the level
// (AMR) index space. damn moves items between patches through the user's free functions,
// found by ADL:
//
//   Box<int, dim> domain_box(resource, patch_box)     // the cells the patch owns
//   Box<int, dim> ghost_box(resource, patch_box)      // the cells it may hold items in
//   void for_each_item(resource const&, fn)           // fn(item const&)
//   Point<int, dim> cell(item)                        // the item's cell
//   void append(resource&, item, shift)               // adds item, its cell moved by shift
//   resource::value_type                              // the item type
//
// items must be trivially copyable; as for BoxData the boxes depend only on the patch box

#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"

#include <cstddef>
#include <utility>
#include <concepts>

namespace damn::amr
{
namespace detail
{
    struct AnyItem
    {
        void operator()(auto const&) const {}
    };
} // namespace detail

template<typename R, std::size_t dim>
concept ParticleData = requires(R& r, R const& cr, core::Box<int, dim> const& box,
                                core::Point<int, dim> const& shift) {
    { domain_box(cr, box) } -> std::same_as<core::Box<int, dim>>;
    { ghost_box(cr, box) } -> std::same_as<core::Box<int, dim>>;
    typename R::value_type;
    for_each_item(cr, detail::AnyItem{});
    { cell(std::declval<typename R::value_type const&>()) } -> std::same_as<core::Point<int, dim>>;
    append(r, std::declval<typename R::value_type const&>(), shift);
};

} // namespace damn::amr

#endif // DAMN_AMR_PARTICLE_DATA_HPP
