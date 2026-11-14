#ifndef DAMN_AMR_TIME_REFINEMENT_HPP
#define DAMN_AMR_TIME_REFINEMENT_HPP

// advancing a hierarchy with time refinement (subcycling, Berger & Colella): each level takes
// ratio steps per step of the next coarser one, after it, then is synchronised onto it
//
//   advance(h, hooks, t, dt): one step dt of level 0 from t, its finer levels recursively
//
// the user's hooks, per level ln (all but advance optional, see below):
//   hooks.advance(ln, t, dt)               one step of level ln, from t to t + dt
//   hooks.prepare_step(ln, t)              before each step of ln, e.g. keep the data at t
//                                          (the next finer level interpolates between it and
//                                          the data at t + dt)
//   hooks.first_step(ln, t, coarse_t, coarse_dt)
//                                          before the first of ln's substeps (ln > 0): the
//                                          coarser level is at coarse_t + coarse_dt already,
//                                          and was at coarse_t
//   hooks.last_step(ln)                    after the last of ln's substeps (ln > 0)
//   hooks.synchronize(ln, t)               after ln's substeps (ln > 0), its data onto level
//                                          ln - 1, both at t
//
// per step of level ln: first_step (first substep), prepare_step, advance, last_step (last
// substep), then the finer level's ratio substeps and its synchronize

#include "damn/amr/hierarchy.hpp"

#include <cstddef>

namespace damn::amr
{
namespace detail
{
    template<std::size_t dim>
    void advance_level(Hierarchy<dim> const& h, auto& hooks, int const ln, double const t,
                       double const dt, bool const first, bool const last, double const coarse_t,
                       double const coarse_dt)
    {
        if constexpr (requires { hooks.first_step(ln, t, coarse_t, coarse_dt); })
            if (ln > 0 and first)
                hooks.first_step(ln, t, coarse_t, coarse_dt);
        if constexpr (requires { hooks.prepare_step(ln, t); })
            hooks.prepare_step(ln, t);

        hooks.advance(ln, t, dt);

        if constexpr (requires { hooks.last_step(ln); })
            if (ln > 0 and last)
                hooks.last_step(ln);

        if (!level_exists(h, ln + 1))
            return;
        double const fine_dt = dt / h.ratio;
        for (int k = 0; k < h.ratio; ++k)
            advance_level(h, hooks, ln + 1, t + k * fine_dt, fine_dt, k == 0, k == h.ratio - 1, t,
                          dt);
        if constexpr (requires { hooks.synchronize(ln + 1, t + dt); })
            hooks.synchronize(ln + 1, t + dt);
    }
} // namespace detail

// one step dt of level 0 from t, with ratio substeps per level per step of the coarser one
template<std::size_t dim>
void advance(Hierarchy<dim> const& h, auto& hooks, double const t, double const dt)
{
    detail::advance_level(h, hooks, 0, t, dt, true, true, t, dt);
}

} // namespace damn::amr

#endif // DAMN_AMR_TIME_REFINEMENT_HPP
