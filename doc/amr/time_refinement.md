# Time refinement

`inc/damn/amr/time_refinement.hpp` advances a hierarchy with time refinement (subcycling, Berger
& Colella): each finer level steps from its coarser level's time to the end of that level's step,
then is synchronized onto it.

How a coarser step is split into substeps follows SAMRAI's `TimeRefinementIntegrator`
(`advanceRecursivelyForRefinedTimestepping`, `findNextDtAndStepsRemaining`).

## Use

```cpp
TimeRefinement<dim> stepping{{.time_ratio = 4, .regrid_interval = 2}};
stepping.advance(h, hooks, t, dt); // one step dt of level 0 from t, finer levels with it
advance(h, hooks, t, dt, time_ratio); // the same with a fresh state (every level new)
stepping.state() / restore(state); // for restarts
```

The state carried across steps is each level's `dt_max` and the finest level before the last
regrid; everything else is reset per coarser step.

### Options

| Option | Meaning |
|---|---|
| `time_ratio` | a finer level's default dt is the coarser one's over this; 0: the refinement ratio (use ratio^2 for dt ~ dx^2) |
| `regrid_interval` | a refinable level's substeps per coarser step are a multiple of this |
| `grow_dt` | `dt_max` grows by at most this factor per step (SAMRAI's default is 1; damn's 1e3) |

## Substeps

After each substep the level asks for a dt: `hooks.next_dt`, otherwise the coarser level's dt over
`time_ratio`. Then:

1. `dt_max = min(asked, dt_max * grow_dt)`;
2. the steps left to the coarser level's end are `n = int(remaining / dt_max)`, plus one if that
   leaves more than `sqrt(eps)` of `remaining`;
3. on a level that can be refined, `n` is rounded up to a multiple of `regrid_interval`;
4. each substep is `remaining / (steps left)`, ending at `t + dt`; the last one ends exactly at
   the coarser level's end.

A level that is new since the previous advance (made by a regrid, or at the first advance) starts
with `dt_max = min(coarser dt, coarser dt / time_ratio)` and, if it can be refined, at most
`remaining / regrid_interval`.

## Hooks

The user's hooks, per level `ln`. All but `advance` are optional.

| Hook | When / what |
|---|---|
| `advance(ln, step)` | one step of level `ln`; `step` is a `Step` (`t`, `dt`, `coarse_t`, `coarse_dt`, `first`, `last`, `substep`) |
| `advance(ln, t, dt)` | the same with only its `t` and `dt` |
| `next_dt(ln, step) -> double` | after that step: the dt the level asks for next (its stable dt, a fixed fraction, ...); default: the coarser level's dt / `time_ratio` (level 0: `step.dt`) |
| `prepare_step(ln, t)` | before each step of `ln`, e.g. to keep the data at `t`: the next finer level interpolates between it and the data at `t + dt` |
| `first_step(ln, t, coarse_t, coarse_dt)` | before the first of `ln`'s substeps (`ln > 0`): the coarser level is already at `coarse_t + coarse_dt`, and was at `coarse_t` |
| `last_step(ln)` | after the last of `ln`'s substeps (`ln > 0`) |
| `synchronize(ln, t)` | after `ln`'s substeps (`ln > 0`): its data onto level `ln - 1`, both at `t` |
| `after_step(ln, t, last)` | after a step of `ln` to `t` with the finer levels (synchronized), the finest included: e.g. a regrid of the finer levels (SAMRAI's regrid points); `last`: `ln`'s last substep of the coarser level's step (true for level 0) |

### Order

Per step of level `ln`:

1. `first_step` (first substep only)
2. `prepare_step`
3. `advance`
4. `next_dt`
5. `last_step` (last substep only)
6. the finer level's substeps, then its `synchronize`
7. `after_step`

A level made by `after_step`'s regrid is new at its first advance: the finest level is recorded
before `after_step` runs (SAMRAI's `d_last_finest_level` at a regrid point).
