#ifndef DAMN_AMR_TIME_REFINEMENT_HPP
#define DAMN_AMR_TIME_REFINEMENT_HPP

// advancing a hierarchy with time refinement (subcycling, Berger & Colella): each finer level
// steps from its coarser level's time to the end of that level's step, then is synchronised onto
// it. How a coarser step is split into substeps is SAMRAI's TimeRefinementIntegrator's
// (advanceRecursivelyForRefinedTimestepping, findNextDtAndStepsRemaining):
//
//   after each substep the level asks for a dt (hooks.next_dt, else the coarser level's dt over
//   time_ratio); dt_max = min(asked, dt_max * grow_dt); the steps left to the coarser level's
//   end, n = int(remaining / dt_max) (+1 if it leaves more than sqrt(eps) of remaining), rounded
//   up to a multiple of regrid_interval on a level that can be refined; each substep
//   remaining / (steps left), its end time t + dt, the last one's the coarser level's end
//   a level new since the previous advance (made by a regrid, or the first advance): its first
//   dt_max min(coarser dt, coarser dt / time_ratio) and, if it can be refined,
//   remaining / regrid_interval
//
//   TimeRefinement<dim> stepping{{.time_ratio = 4, .regrid_interval = 2}};
//   stepping.advance(h, hooks, t, dt);     one step dt of level 0 from t, finer levels with it
//   advance(h, hooks, t, dt, time_ratio)   the same with a fresh state (every level new)
//
// the user's hooks, per level ln (all but advance optional, see below):
//   hooks.advance(ln, step)                one step of level ln, step: a Step (t, dt, coarse_t,
//                                          coarse_dt, first, last, substep); or
//   hooks.advance(ln, t, dt)               its t and dt only
//   hooks.next_dt(ln, step) -> double      after that step: the dt the level asks for next (its
//                                          stable dt, a fixed fraction...); default: the
//                                          coarser level's dt / time_ratio (level 0: step.dt)
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
//   hooks.after_step(ln, t, last)          after a step of ln to t and the finer levels with it
//                                          (synchronized), finest level included: e.g. a regrid
//                                          of the finer levels (SAMRAI's regrid points); last:
//                                          ln's last substep of the coarser level's step (true
//                                          for level 0)
//
// per step of level ln: first_step (first substep), prepare_step, advance, next_dt, last_step
// (last substep), then the finer level's substeps and its synchronize, then after_step

#include "damn/amr/hierarchy.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace damn::amr
{

// one step of a level, for hooks.advance / hooks.next_dt
struct Step
{
    double t         = 0;     // from
    double dt        = 0;     // to t + dt (the last substep: exactly the coarser level's end)
    double coarse_t  = 0;     // the coarser level's step: from coarse_t (level 0: t)
    double coarse_dt = 0;     //  to coarse_t + coarse_dt (level 0: dt)
    bool first       = false; // the first substep in the coarser level's step (level 0: true)
    bool last        = false; // the last one (level 0: true)
    int substep      = 0;     // its index in the coarser level's step
};

struct TimeRefinementOptions
{
    int time_ratio = 0;        // a finer level's default dt: the coarser one's over this (0:
                               //  the refinement ratio; ratio^2 for dt ~ dx^2)
    int regrid_interval = 1;   // a refinable level's substeps: a multiple of this
    double grow_dt      = 1e3; // dt_max grows by at most this factor per step (SAMRAI's 1)
};

namespace detail
{
    // hooks.advance: with the Step, or its t and dt
    void advance_hook(auto& hooks, int const ln, Step const& step)
        requires requires { hooks.advance(ln, step); }
    {
        hooks.advance(ln, step);
    }
    void advance_hook(auto& hooks, int const ln, Step const& step)
        requires(!requires { hooks.advance(ln, step); })
    {
        hooks.advance(ln, step.t, step.dt);
    }

    // hooks.next_dt, else the default bound
    double next_dt_hook(auto& hooks, int const ln, Step const& step,
                        [[maybe_unused]] int const time_ratio)
        requires requires {
            { hooks.next_dt(ln, step) } -> std::convertible_to<double>;
        }
    {
        return hooks.next_dt(ln, step);
    }
    double next_dt_hook(auto&, int const ln, Step const& step, int const time_ratio)
    {
        return ln == 0 ? step.dt : step.coarse_dt / time_ratio;
    }
} // namespace detail

template<std::size_t dim>
class TimeRefinement
{
public:
    explicit TimeRefinement(TimeRefinementOptions const& options = {})
        : options_{options}
    {
    }

    // one step dt of level 0 from t, its finer levels with it
    void advance(Hierarchy<dim> const& h, auto& hooks, double const t, double const dt)
    {
        ratio_ = options_.time_ratio > 0 ? options_.time_ratio : h.ratio;
        if (levels_.size() < static_cast<std::size_t>(std::max(h.max_levels, 1)))
            levels_.resize(static_cast<std::size_t>(std::max(h.max_levels, 1)));
        auto& l0         = levels_[0];
        l0.sim_time      = t;
        level0_advanced_ = false;
        if (!(l0.dt_max > 0))
            l0.dt_max = dt;
        advance_level_(h, hooks, 0, t + dt, t, dt);
    }

private:
    struct LevelState
    {
        double dt_max    = 0; // the largest dt it may take
        double dt_actual = 0; // its next substep's
        double sim_time  = 0; // its time
        int step         = 0; // its substeps done in the coarser level's step
        int max_steps    = 1; // and to do
    };

    TimeRefinementOptions options_;
    std::vector<LevelState> levels_;
    int ratio_            = 2;
    int last_finest_      = 0; // the finest level before the last regrid (SAMRAI's)
    bool level0_advanced_ = false;

    static bool can_be_refined_(Hierarchy<dim> const& h, int const ln)
    {
        return ln + 1 < h.max_levels;
    }

    bool last_level_step_(int const ln) const
    {
        auto const& l = levels_[static_cast<std::size_t>(ln)];
        return ln == 0 ? level0_advanced_ : l.step >= l.max_steps;
    }

    // SAMRAI's findNextDtAndStepsRemaining: dt_max, the steps left, the next substep's dt;
    //  whether that one is the last
    bool next_dt_and_steps_(Hierarchy<dim> const& h, int const ln, double const time_remaining,
                            double const dt_bound)
    {
        auto& l  = levels_[static_cast<std::size_t>(ln)];
        l.dt_max = std::min(dt_bound, l.dt_max * options_.grow_dt);

        bool const steps_remaining = ln == 0 ? !level0_advanced_ : l.step < l.max_steps;
        if (steps_remaining)
        {
            auto n = static_cast<int>(time_remaining / l.dt_max);
            if (time_remaining - l.dt_max * n
                > std::sqrt(std::numeric_limits<double>::epsilon()) * time_remaining)
                ++n;
            if (ln > 0)
                l.max_steps = l.step + n;
            if (ln > 0 and can_be_refined_(h, ln))
            {
                auto const interval = std::max(options_.regrid_interval, 1);
                l.max_steps         = std::max(l.max_steps, interval);
                auto regrids        = l.max_steps / interval;
                if (l.max_steps % interval)
                    ++regrids;
                l.max_steps = regrids * interval;
                if (l.step >= l.max_steps)
                    throw std::runtime_error("damn::amr::TimeRefinement: no steps left on level "
                                             + std::to_string(ln));
            }
            l.dt_actual = ln == 0 ? time_remaining
                                  : time_remaining / static_cast<double>(l.max_steps - l.step);
        }
        else
            l.dt_actual = l.dt_max;

        return ln == 0 ? true : (l.max_steps - l.step) <= 1;
    }

    // SAMRAI's advanceRecursivelyForRefinedTimestepping: level ln from its coarser level's time
    //  (coarse_t) to end_time (= coarse_t + coarse_dt)
    void advance_level_(Hierarchy<dim> const& h, auto& hooks, int const ln, double const end_time,
                        double const coarse_t, double const coarse_dt)
    {
        auto& l               = levels_[static_cast<std::size_t>(ln)];
        double time_remaining = 0;
        if (ln > 0)
        {
            auto const& coarse = levels_[static_cast<std::size_t>(ln) - 1];
            l.step             = 0;
            l.max_steps        = 1;
            l.sim_time         = coarse.sim_time;
            time_remaining     = end_time - l.sim_time;
            if (last_finest_ < ln) // new since the last regrid: bounded by the coarser level
            {
                l.dt_max = std::min(coarse.dt_actual, coarse.dt_actual / ratio_);
                if (can_be_refined_(h, ln))
                    l.dt_max = std::min(
                        l.dt_max, time_remaining
                                      / static_cast<double>(std::max(options_.regrid_interval, 1)));
                last_finest_ = ln;
            }
        }
        else
            time_remaining = end_time - l.sim_time;

        bool last = next_dt_and_steps_(h, ln, time_remaining, l.dt_max);

        while (!last_level_step_(ln))
        {
            double const new_time = last ? end_time : l.sim_time + l.dt_actual;
            Step const step{.t         = l.sim_time,
                            .dt        = new_time - l.sim_time,
                            .coarse_t  = ln == 0 ? l.sim_time : coarse_t,
                            .coarse_dt = ln == 0 ? new_time - l.sim_time : coarse_dt,
                            .first     = ln == 0 or l.step == 0,
                            .last      = last,
                            .substep   = ln == 0 ? 0 : l.step};

            if constexpr (requires { hooks.first_step(ln, step.t, coarse_t, coarse_dt); })
                if (ln > 0 and step.first)
                    hooks.first_step(ln, step.t, coarse_t, coarse_dt);
            if constexpr (requires { hooks.prepare_step(ln, step.t); })
                hooks.prepare_step(ln, step.t);

            detail::advance_hook(hooks, ln, step);
            double const dt_new = detail::next_dt_hook(hooks, ln, step, ratio_);

            if constexpr (requires { hooks.last_step(ln); })
                if (ln > 0 and last)
                    hooks.last_step(ln);

            if (ln == 0)
                level0_advanced_ = true;
            else
                ++l.step;

            if (level_exists(h, ln + 1))
            {
                advance_level_(h, hooks, ln + 1, new_time, step.t, step.dt);
                if constexpr (requires { hooks.synchronize(ln + 1, new_time); })
                    hooks.synchronize(ln + 1, new_time);
            }

            // the finest level before after_step's regrid: a level it makes is new at its first
            //  advance (SAMRAI: d_last_finest_level at a regrid point)
            int const finest = finest_level(h);
            if constexpr (requires { hooks.after_step(ln, new_time, last); })
                hooks.after_step(ln, new_time, last);
            last_finest_ = finest;

            l.sim_time = new_time;
            last       = next_dt_and_steps_(h, ln, end_time - new_time, dt_new);
        }
    }
};

// one step dt of level 0 from t with a fresh stepping state (every level new: each finer level
//  time_ratio substeps per step of its coarser level when no hooks.next_dt), 0: the refinement
//  ratio
template<std::size_t dim>
void advance(Hierarchy<dim> const& h, auto& hooks, double const t, double const dt,
             int const time_ratio = 0)
{
    TimeRefinement<dim> stepping{{.time_ratio = time_ratio}};
    stepping.advance(h, hooks, t, dt);
}

} // namespace damn::amr

#endif // DAMN_AMR_TIME_REFINEMENT_HPP
