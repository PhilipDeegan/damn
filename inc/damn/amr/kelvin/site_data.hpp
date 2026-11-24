#ifndef DAMN_AMR_KELVIN_SITE_DATA_HPP
#define DAMN_AMR_KELVIN_SITE_DATA_HPP

// values on a Kelvin lattice's sites, as damn BoxData: what a model keeps per site (a scalar, a
// value per face, any trivially copyable T), ghosts included, moved by damn like any field
//
//   SiteArray<T, dim>   one sublattice (corner or body) of a patch: a T per cell over the patch's
//                       cells grown by ghost_width, C order (box_span: copies go by rows)
//   SiteField<T, dim>   both sublattices
//   for_each_site_resource(name, field, fn)   fn(name + ".corner", field.corner),
//                       fn(name + ".body", field.body): a PatchData's for_each_resource
//   site_names({names})  every name's two, for copy(fill_ghosts, neighbours, names, ld)
//
// a site absent from the domain (KeyBox::has, a bounded edge) still has a slot: models leave it be

#include "damn/core/def.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/memory/box_span.hpp"
#include "damn/amr/box_data.hpp"
#include "damn/amr/lattice.hpp"

#include <string>
#include <vector>
#include <cassert>
#include <cstddef>
#include <string_view>
#include <type_traits>
#include <initializer_list>

namespace damn::core
{
template<typename Type, std::size_t dim>
class Point;
} // namespace damn::core

namespace damn::amr::kelvin
{
template<typename T, std::size_t dim>
struct SiteArray
{
    static_assert(std::is_trivially_copyable_v<T>);
    using value_type = T;

    Site site       = Site::corner;
    int ghost_width = 1;
    std::vector<T> v{};

    // storage for the patch of cells, every value init
    void allocate(core::Box<int, dim> const& cells, T const& init = T{});
};

template<typename T, std::size_t dim>
struct SiteField
{
    SiteArray<T, dim> corner{Site::corner}, body{Site::body};

    void allocate(core::Box<int, dim> const& cells, T const& init = T{});
    MKN_KUL_FN_SIG_RET SiteArray<T, dim>& operator[](Site site);
    MKN_KUL_FN_SIG_RET SiteArray<T, dim> const& operator[](Site site) const;
};

template<typename T, std::size_t dim>
MKN_KUL_FN_SIG_RET core::Box<int, dim> domain_box(SiteArray<T, dim> const& a,
                                                  core::Box<int, dim> const& cells);
template<typename T, std::size_t dim>
MKN_KUL_FN_SIG_RET core::Box<int, dim> ghost_box(SiteArray<T, dim> const& a,
                                                 core::Box<int, dim> const& cells);
template<typename T, std::size_t dim>
MKN_KUL_FN_SIG_RET T& value(SiteArray<T, dim>& a, core::Box<int, dim> const& cells,
                            core::Point<int, dim> const& cell);
template<typename T, std::size_t dim>
MKN_KUL_FN_SIG_RET T const& value(SiteArray<T, dim> const& a, core::Box<int, dim> const& cells,
                                  core::Point<int, dim> const& cell);
template<typename T, std::size_t dim>
MKN_KUL_FN_SIG_RET core::BoxSpan<T, dim> box_span(SiteArray<T, dim>& a,
                                                  core::Box<int, dim> const& cells);
template<typename T, std::size_t dim>
MKN_KUL_FN_SIG_RET core::BoxSpan<T const, dim> box_span(SiteArray<T, dim> const& a,
                                                        core::Box<int, dim> const& cells);

template<typename T, std::size_t dim>
void for_each_site_resource(std::string_view name, SiteField<T, dim>& field, auto&& fn);

MKN_KUL_FN_SIG_RET std::vector<std::string>
site_names(std::initializer_list<std::string_view> names);

} // namespace damn::amr::kelvin


namespace damn::amr::kelvin
{
template<typename T, std::size_t dim>
void SiteArray<T, dim>::allocate(core::Box<int, dim> const& cells, T const& init)
{
    v.assign(ghost_box(*this, cells).size(), init);
}

template<typename T, std::size_t dim>
void SiteField<T, dim>::allocate(core::Box<int, dim> const& cells, T const& init)
{
    corner.allocate(cells, init);
    body.allocate(cells, init);
}

template<typename T, std::size_t dim>
SiteArray<T, dim>& SiteField<T, dim>::operator[](Site const site)
{
    return site == Site::body ? body : corner;
}

template<typename T, std::size_t dim>
SiteArray<T, dim> const& SiteField<T, dim>::operator[](Site const site) const
{
    return site == Site::body ? body : corner;
}

template<typename T, std::size_t dim>
core::Box<int, dim> domain_box(SiteArray<T, dim> const&, core::Box<int, dim> const& cells)
{
    return cells;
}

template<typename T, std::size_t dim>
core::Box<int, dim> ghost_box(SiteArray<T, dim> const& a, core::Box<int, dim> const& cells)
{
    return core::grow(cells, a.ghost_width);
}

template<typename T, std::size_t dim>
core::BoxSpan<T, dim> box_span(SiteArray<T, dim>& a, core::Box<int, dim> const& cells)
{
    assert(a.v.size() == ghost_box(a, cells).size());
    return {a.v.data(), ghost_box(a, cells)};
}

template<typename T, std::size_t dim>
core::BoxSpan<T const, dim> box_span(SiteArray<T, dim> const& a, core::Box<int, dim> const& cells)
{
    assert(a.v.size() == ghost_box(a, cells).size());
    return {a.v.data(), ghost_box(a, cells)};
}

template<typename T, std::size_t dim>
T& value(SiteArray<T, dim>& a, core::Box<int, dim> const& cells, core::Point<int, dim> const& cell)
{
    assert(core::isIn(cell, ghost_box(a, cells)));
    return box_span(a, cells)(cell);
}

template<typename T, std::size_t dim>
T const& value(SiteArray<T, dim> const& a, core::Box<int, dim> const& cells,
               core::Point<int, dim> const& cell)
{
    assert(core::isIn(cell, ghost_box(a, cells)));
    return box_span(a, cells)(cell);
}

template<typename T, std::size_t dim>
void for_each_site_resource(std::string_view const name, SiteField<T, dim>& field, auto&& fn)
{
    fn(std::string{name} + ".corner", field.corner);
    fn(std::string{name} + ".body", field.body);
}

inline std::vector<std::string> site_names(std::initializer_list<std::string_view> const names)
{
    std::vector<std::string> out;
    out.reserve(2 * names.size());
    for (auto const name : names)
    {
        out.emplace_back(std::string{name} + ".corner");
        out.emplace_back(std::string{name} + ".body");
    }
    return out;
}

static_assert(BoxData<SiteArray<double, 3>, 3>);

} // namespace damn::amr::kelvin

#endif // DAMN_AMR_KELVIN_SITE_DATA_HPP
