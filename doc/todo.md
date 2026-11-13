# TODO: move the initializer out of damn

damn moves data around and does not know what the data is. The input layer
(`damn/initializer/data_provider.hpp`) is plamn's, so it moves there, and the
static dict handler goes away.

## Current state

- `DAMNDictHandler::INSTANCE()` (singleton) and `DataProvider` are unused: the
  only code touching them is their definition, `src/damn/initializer/data_provider.cpp`.
- `damn::initializer::DAMNDict` and `damn::initializer::InitFunction<dim>` are
  used only by plamn: the constructors of `Ions`, `IonPopulation`, `Electromag`,
  `Electrons` (and its closures), `VecFieldInitializer`, `SimOpts::FROM`,
  `OhmInfo::FROM`, `ParticleInitializerFactory`, `MaxwellianParticleInitializer`,
  plus the plamn tests that build a dict (`test_updater.cpp`, ohm `test_main.cpp`).
- After that, damn's only cppdict use is `DictionaryException` in
  `damn/core/errors.hpp`, thrown and caught only by plamn
  (`numerics/pusher/boris.hpp`, `numerics/ion_updater/ion_updater.hpp`).

## damn (this PR)

- [x] Delete `inc/damn/initializer/` and `src/damn/initializer/`.
- [x] Move `DictionaryException` out of `inc/damn/core/errors.hpp` (to plamn).
- [x] Drop cppdict: the `sub:` and `inc:` lines in `mkn.yaml` `base`.
- [x] `damn_core` then compiles only `tpp/phlop/src/phlop/timing/scope_timer.cpp`.
- [x] Remove the `damn/initializer/...` include from `inc/damn/damn_core.hpp`
      (it is generated from the header list, so just drop the line).
- [x] Build and test from a clean clone (`rm -rf tpp bin .mkn`), as CI does.

## plamn (companion PR, merge after this one)

- [x] Add `inc/plamn/initializer/data_provider.hpp` with `namespace plamn::initializer`:
      the dict type (rename `DAMNDict`, e.g. `Dict`), `InitFunction<dim>` and
      `InitFunctionHelper`, the `cppdict::get_value` enum overload. No handler, no
      singleton, no `DataProvider`.
- [x] Add `DictionaryException` to plamn (e.g. `plamn/core/errors.hpp`).
- [x] Replace every `damn::initializer::` and `damn::core::DictionaryException`
      use with the plamn names; include the new headers.
- [x] Add cppdict to plamn's `mkn.yaml` `base` (`sub:` + `inc:`), as damn had it.
- [x] Build and test (`mkn clean build test -p test_plamn_core -dKO -t 8`):
      expect 2/30/69/4/2 passing.

## Later: Python owns the dict

The dict is passed by `const&` everywhere, so a Python object (nanobind, when
plamn gets bindings) can own it for its lifetime and hand it to the simulator.
No static handler is needed for that.
