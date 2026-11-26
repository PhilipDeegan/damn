# Bootstrap facts (`def/bootstrap.hpp`)

What this machine and build can do, found by `etc/bootstrap/bootstrap.py`: its toolchain
discovery (`discover.py`), then one probe binary per capability, built, run and checked.

The result is `damn/core/def/_gen_bootstrap.hpp`, generated and not committed. Without it,
`bootstrap_defaults.hpp` (committed, every capability off, made by `bootstrap.py --defaults`) is
included instead.

A run still checks for itself what it uses (another node, the MPI thread level): the facts are
the build machine's.

## Compile time

| Name | What |
|---|---|
| `DAMN_HAVE_BOOTSTRAP`, `DAMN_HAVE_<AREA>_<PROBE>` | 1 / 0, for `#if` |
| `damn::bootstrap::Toolchain::<flag>` | `discover.py`'s flags (bool): `mpi_cuda_aware`, `hdf5_subfiling`, `cuda_found`, ...; `gpu`: `"none"`, `"cuda"` or `"rocm"` |
| `damn::bootstrap::<Area>::<probe>` | each probe passed (`Mpi::thread_multiple`, ...) |

## Run time

Every fact is a string, by path.

| Name | What |
|---|---|
| `damn::bootstrap::entries` | `(path, value)` pairs; paths are `<area>/<probe>/<key>`, `<area>/<probe>/ok`, `toolchain/<area>/<key>` and `settings`. Empty without a bootstrap |
| `damn::bootstrap::get<T>(path, fallback)` | a fact as `T` (kul's `from_string`), `fallback` without it, e.g. `get<double>("mpi/exchange_node/latency_s", 1e-6)` |
| `damn::bootstrap::tree()` | the facts as a bon tree (`damn/core/bon.hpp`): `bon::find(tree(), "mpi")` for an area's |
