#ifndef DAMN_AMR_KELVIN_INTER_LEVEL_HPP
#define DAMN_AMR_KELVIN_INTER_LEVEL_HPP

// a Kelvin lattice's sites between levels at an odd ratio, each sublattice nesting in itself,
// and the plain refine / coarsen operators: doc/amr/kelvin/inter_level.md

#include "damn/core/def.hpp"
#include "damn/core/geometry/box.hpp"
#include "damn/core/geometry/box_spans.hpp"
#include "damn/core/geometry/point.hpp"

#include "damn/amr/kelvin/site_data.hpp"
#include "damn/amr/kelvin/sites.hpp"
#include "damn/amr/lattice.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace damn::amr
{
template<typename T, std::size_t dim>
struct BoxValues;
} // namespace damn::amr

namespace damn::amr::kelvin
{
template<std::size_t dim>
struct Bracket
{
    core::Point<int, dim> lower{}; // coarse cell c per direction
    std::array<int, dim> num{};    // the fine site at c + num / r, 0 <= num < r
};

// the offset in fine cells of a site's coincident fine cell from r c: 0 corner, (r - 1) / 2 body
MKN_KUL_FN_SIG_RET constexpr int nest_offset(Site site, int ratio);

template<std::size_t dim>
MKN_KUL_FN_SIG_RET core::Point<int, dim> coincident_fine(core::Point<int, dim> const& coarse_cell,
                                                         Site site, int ratio);

template<std::size_t dim>
MKN_KUL_FN_SIG_RET std::optional<core::Point<int, dim>>
coincident_coarse(core::Point<int, dim> const& fine_cell, Site site, int ratio);

template<std::size_t dim>
MKN_KUL_FN_SIG_RET Bracket<dim> bracket(core::Point<int, dim> const& fine_cell, Site site,
                                        int ratio);

// keys r times (a key K is the fine key r K, its own sublattice's); a periodic direction's last
//  cell r cells: its upper key r - 1 more
template<std::size_t dim>
MKN_KUL_FN_SIG_RET KeyBox<dim> refined_keys(KeyBox<dim> const& keys, int ratio);

void check_ratio(int ratio); // throws if not odd and positive

struct LinearRefine
{
    int ratio   = 3;
    int stencil = 1;

    template<typename T, std::size_t dim>
    void operator()(SiteArray<T, dim>& fine, core::Box<int, dim> const& fine_patch_box,
                    core::Box<int, dim> const& box, BoxValues<T const, dim> const& coarse) const;
};

struct InjectCoarsen
{
    int ratio = 3;

    template<typename T, std::size_t dim>
    void operator()(SiteArray<T, dim> const& fine, core::Box<int, dim> const& fine_patch_box,
                    BoxValues<T, dim> const& out) const;
};

} // namespace damn::amr::kelvin


namespace damn::amr::kelvin
{
inline void check_ratio(int const ratio)
{
    if (ratio < 1 or ratio % 2 == 0)
        throw std::runtime_error("damn::amr::kelvin: ratio " + std::to_string(ratio)
                                 + ": a Kelvin lattice nests at odd ratios only");
}

constexpr int nest_offset(Site const site, int const ratio)
{
    return site == Site::body ? (ratio - 1) / 2 : 0;
}

template<std::size_t dim>
core::Point<int, dim> coincident_fine(core::Point<int, dim> const& coarse_cell, Site const site,
                                      int const ratio)
{
    core::Point<int, dim> f;
    for (std::size_t d = 0; d < dim; ++d)
        f[d] = ratio * coarse_cell[d] + nest_offset(site, ratio);
    return f;
}

template<std::size_t dim>
std::optional<core::Point<int, dim>> coincident_coarse(core::Point<int, dim> const& fine_cell,
                                                       Site const site, int const ratio)
{
    auto const b = bracket(fine_cell, site, ratio);
    for (auto const n : b.num)
        if (n != 0)
            return std::nullopt;
    return b.lower;
}

template<std::size_t dim>
Bracket<dim> bracket(core::Point<int, dim> const& fine_cell, Site const site, int const ratio)
{
    Bracket<dim> b;
    for (std::size_t d = 0; d < dim; ++d)
    {
        auto const x = fine_cell[d] - nest_offset(site, ratio);
        b.lower[d]   = core::floor_div(x, ratio);
        b.num[d]     = x - ratio * b.lower[d];
    }
    return b;
}

template<std::size_t dim>
KeyBox<dim> refined_keys(KeyBox<dim> const& keys, int const ratio)
{
    check_ratio(ratio);
    auto fine = keys;
    for (std::size_t d = 0; d < dim; ++d)
    {
        fine.lower[d] = ratio * keys.lower[d];
        fine.upper[d] = ratio * keys.upper[d] + (keys.periodic[d] ? ratio - 1 : 0);
    }
    return fine;
}


namespace detail
{
    // out += w v, for a floating point or an array of them
    template<typename T>
    void add_scaled(T& out, double const w, T const& v)
    {
        if constexpr (std::is_floating_point_v<T>)
            out += static_cast<T>(w) * v;
        else
            for (std::size_t i = 0; i < out.size(); ++i)
                add_scaled(out[i], w, v[i]);
    }
} // namespace detail

template<typename T, std::size_t dim>
void LinearRefine::operator()(SiteArray<T, dim>& fine, core::Box<int, dim> const& fine_patch_box,
                              core::Box<int, dim> const& box,
                              BoxValues<T const, dim> const& coarse) const
{
    check_ratio(ratio);
    core::for_each_point(box, [&](auto const& f) {
        auto const b = bracket(f, fine.site, ratio);
        T v{};
        for (std::size_t corner = 0; corner < (std::size_t{1} << dim); ++corner)
        {
            double w = 1;
            auto c   = b.lower;
            for (std::size_t d = 0; d < dim; ++d)
            {
                auto const t = static_cast<double>(b.num[d]) / ratio;
                if ((corner >> d) & 1)
                {
                    w *= t;
                    ++c[d];
                }
                else
                    w *= 1 - t;
            }
            if (w != 0)
                detail::add_scaled(v, w, coarse(c));
        }
        value(fine, fine_patch_box, f) = v;
    });
}

template<typename T, std::size_t dim>
void InjectCoarsen::operator()(SiteArray<T, dim> const& fine,
                               core::Box<int, dim> const& fine_patch_box,
                               BoxValues<T, dim> const& out) const
{
    check_ratio(ratio);
    core::for_each_point(out.box, [&](auto const& c) {
        out(c) = value(fine, fine_patch_box, coincident_fine(c, fine.site, ratio));
    });
}

} // namespace damn::amr::kelvin

#endif // DAMN_AMR_KELVIN_INTER_LEVEL_HPP
