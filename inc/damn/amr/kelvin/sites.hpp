#ifndef DAMN_AMR_KELVIN_SITES_HPP
#define DAMN_AMR_KELVIN_SITES_HPP

// a Kelvin (body centred) lattice's sites on damn's cells, keys in h units, and their faces:
// doc/amr/kelvin/sites.md
// face order: 2^dim diagonal faces, signs lexicographic from (-,-,-), then 2 dim axial faces
//  -x, +x, -y, +y, ...

#include "damn/core/def.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/point.hpp"
#include "damn/amr/lattice.hpp"

#include <array>
#include <cstddef>
#include <utility>
#include <stdexcept>

namespace damn::amr::kelvin
{
template<std::size_t dim>
using Key = core::Point<int, dim>;

template<std::size_t dim>
std::size_t constexpr n_diagonal_faces = std::size_t{1} << dim;

template<std::size_t dim>
std::size_t constexpr n_faces = n_diagonal_faces<dim> + 2 * dim;

MKN_KUL_FN_SIG_RET constexpr int body_offset(Site site);

template<std::size_t dim>
MKN_KUL_FN_SIG_RET Key<dim> key_of(core::Point<int, dim> const& cell, Site site);

template<std::size_t dim>
MKN_KUL_FN_SIG_RET core::Point<int, dim> cell_of(Key<dim> const& key);

template<std::size_t dim>
MKN_KUL_FN_SIG_RET Site site_of(Key<dim> const& key);

// face k's step, h units
template<std::size_t dim>
MKN_KUL_FN_SIG_RET constexpr std::array<int, dim> face_delta(std::size_t k);

template<std::size_t dim>
MKN_KUL_FN_SIG_RET constexpr bool is_diagonal(std::size_t k);

// the same face seen from the other side
template<std::size_t dim>
MKN_KUL_FN_SIG_RET constexpr std::size_t mirror(std::size_t k);

template<std::size_t dim>
MKN_KUL_FN_SIG_RET std::pair<core::Point<int, dim>, Site>
neighbour(core::Point<int, dim> const& cell, Site site, std::size_t k);

template<std::size_t dim>
struct KeyBox
{
    Key<dim> lower{}, upper{};
    std::array<bool, dim> periodic{};

    MKN_KUL_FN_SIG_RET bool
    has(Key<dim> const& key) const; // past a bounded end not, periodic images yes
    MKN_KUL_FN_SIG_RET bool has(core::Point<int, dim> const& cell, Site site) const;
    MKN_KUL_FN_SIG_RET core::Box<int, dim> cells() const;
    MKN_KUL_FN_SIG_RET Key<dim> wrap(Key<dim> key) const; // periodic directions into lower..upper
    void check() const; // throws if a periodic direction cuts a cell
};

} // namespace damn::amr::kelvin


namespace damn::amr::kelvin
{
constexpr int body_offset(Site const site)
{
    return site == Site::body ? 1 : 0;
}

template<std::size_t dim>
Key<dim> key_of(core::Point<int, dim> const& cell, Site const site)
{
    Key<dim> key;
    for (std::size_t d = 0; d < dim; ++d)
        key[d] = 2 * cell[d] + body_offset(site);
    return key;
}

template<std::size_t dim>
core::Point<int, dim> cell_of(Key<dim> const& key)
{
    return core::floor_div(key, 2);
}

template<std::size_t dim>
Site site_of(Key<dim> const& key)
{
    return (key[0] & 1) ? Site::body : Site::corner;
}

template<std::size_t dim>
constexpr std::array<int, dim> face_delta(std::size_t const k)
{
    std::array<int, dim> d{};
    if (k < n_diagonal_faces<dim>)
    {
        for (std::size_t i = 0; i < dim; ++i)
            d[i] = (k >> (dim - 1 - i)) & 1 ? 1 : -1;
        return d;
    }
    auto const a = k - n_diagonal_faces<dim>;
    d[a / 2]     = a % 2 ? 2 : -2;
    return d;
}

template<std::size_t dim>
constexpr bool is_diagonal(std::size_t const k)
{
    return k < n_diagonal_faces<dim>;
}

template<std::size_t dim>
constexpr std::size_t mirror(std::size_t const k)
{
    if (is_diagonal<dim>(k))
        return n_diagonal_faces<dim> - 1 - k;
    return k ^ 1u; // n_diagonal_faces is even: -x, +x pairs
}

template<std::size_t dim>
std::pair<core::Point<int, dim>, Site> neighbour(core::Point<int, dim> const& cell, Site const site,
                                                 std::size_t const k)
{
    auto key        = key_of(cell, site);
    auto const step = face_delta<dim>(k);
    for (std::size_t d = 0; d < dim; ++d)
        key[d] += step[d];
    return {cell_of(key), site_of(key)};
}


template<std::size_t dim>
bool KeyBox<dim>::has(Key<dim> const& key) const
{
    for (std::size_t d = 0; d < dim; ++d)
        if (!periodic[d] and (key[d] < lower[d] or key[d] > upper[d]))
            return false;
    return true;
}

template<std::size_t dim>
bool KeyBox<dim>::has(core::Point<int, dim> const& cell, Site const site) const
{
    return has(key_of(cell, site));
}

template<std::size_t dim>
core::Box<int, dim> KeyBox<dim>::cells() const
{
    return {cell_of(lower), cell_of(upper)};
}

template<std::size_t dim>
Key<dim> KeyBox<dim>::wrap(Key<dim> key) const
{
    for (std::size_t d = 0; d < dim; ++d)
        if (periodic[d])
        {
            auto const n = upper[d] - lower[d] + 1;
            key[d]       = ((key[d] - lower[d]) % n + n) % n + lower[d];
        }
    return key;
}

template<std::size_t dim>
void KeyBox<dim>::check() const
{
    for (std::size_t d = 0; d < dim; ++d)
    {
        if (lower[d] > upper[d])
            throw std::runtime_error("damn::amr::kelvin::KeyBox: no keys");
        if (periodic[d] and ((lower[d] & 1) or !(upper[d] & 1)))
            throw std::runtime_error("damn::amr::kelvin::KeyBox: a periodic direction's keys are "
                                     "whole cells (lower even, upper odd)");
    }
}

} // namespace damn::amr::kelvin

#endif // DAMN_AMR_KELVIN_SITES_HPP
