#ifndef DAMN_AMR_PHYSICAL_BOUNDARY_HPP
#define DAMN_AMR_PHYSICAL_BOUNDARY_HPP

// a patch's ghost cells past the physical (non periodic) boundaries of its level's domain, by
//  region: a side per direction (-1 past the lower face, +1 past the upper one, 0 within the
//  domain along it), one non zero side a face's region, two an edge's, three a corner's (SAMRAI's
//  physical boundary boxes). Periodic directions have none: their ghosts are other patches'
//  (periodic images). What fills them (boundary conditions) is the model's

#include "damn/core/def.hpp"
#include "damn/amr/hierarchy.hpp"
#include "damn/core/geometry/box.hpp"

#include "mkn/kul/std/array.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace damn::amr
{
template<std::size_t dim>
struct PhysicalBoundaryRegion
{
    std::array<std::int8_t, dim> sides{}; // per direction: -1 lower, +1 upper, 0 within
    core::Box<int, dim> box{};            // the ghost cells (level index space)

    // 1: a face, 2: an edge, 3: a corner
    MKN_KUL_FN_SIG_RET std::size_t codimension() const
    {
        std::size_t n = 0;
        for (auto const s : sides)
            n += s != 0;
        return n;
    }
};

// the regions of box grown by ghost_width past level ln's physical boundaries
template<std::size_t dim, typename Lattice>
MKN_KUL_FN_SIG_RET std::vector<PhysicalBoundaryRegion<dim>>
physical_boundary_regions(Hierarchy<dim, Lattice> const& h, int const ln,
                          core::Box<int, dim> const& box, int const ghost_width)
{
    auto const domain = level_domain(h, ln);
    auto const ghosts = core::grow(box, mkn::kul::ConstArray<int, dim>(ghost_width));

    std::vector<PhysicalBoundaryRegion<dim>> regions;
    std::size_t combinations = 1;
    for (std::size_t i = 0; i < dim; ++i)
        combinations *= 3;
    for (std::size_t flat = 0; flat < combinations; ++flat)
    {
        PhysicalBoundaryRegion<dim> r;
        bool empty = false;
        auto rest  = flat;
        for (std::size_t i = 0; i < dim and !empty; ++i)
        {
            auto const s = static_cast<std::int8_t>(static_cast<int>(rest % 3) - 1);
            rest /= 3;
            r.sides[i] = s;
            if (s != 0 and h.periodic[i])
                empty = true; // periodic: no physical boundary along i
            else if (s < 0)
            {
                r.box.lower[i] = ghosts.lower[i];
                r.box.upper[i] = domain.lower[i] - 1;
            }
            else if (s > 0)
            {
                r.box.lower[i] = domain.upper[i] + 1;
                r.box.upper[i] = ghosts.upper[i];
            }
            else if (h.periodic[i]) // a periodic image is past the same faces: all its ghosts
            {
                r.box.lower[i] = ghosts.lower[i];
                r.box.upper[i] = ghosts.upper[i];
            }
            else
            {
                r.box.lower[i] = std::max(ghosts.lower[i], domain.lower[i]);
                r.box.upper[i] = std::min(ghosts.upper[i], domain.upper[i]);
            }
            empty = empty or r.box.lower[i] > r.box.upper[i];
        }
        if (!empty and r.codimension() > 0) // sides all 0: the patch itself
            regions.push_back(r);
    }
    return regions;
}

} // namespace damn::amr

#endif // DAMN_AMR_PHYSICAL_BOUNDARY_HPP
