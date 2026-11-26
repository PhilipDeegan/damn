# Conventions

## Comments

A comment says what the code cannot say by itself:

- why: a choice that looks wrong, or a cheaper option that does not work
- edge cases and subtle footguns: what breaks if a reader "simplifies" the code
- workarounds: for which compiler, library or hardware, and when they can go
- optimization strategies: the layout or ordering the code relies on to be fast

Not:

- what the next lines do, or the function name and parameters again
- lists of users, callers or places something is used (`@see`, "used by ..."): grep says it, and
  such lists go stale
- history: "previously ...", "now ...", "fixed by ...", "no longer ...". That belongs in the
  commit and the PR, not in the code

Docstrings are fine when terse and saying more than the signature.

Long explanations (an algorithm, a data flow, a protocol between ranks) go in `doc/` as markdown,
with SVG diagrams where a picture helps; the code points to the doc in one line. `doc/` mirrors
the source tree: `inc/damn/amr/decompose.hpp` is explained in `doc/amr/decompose.md`. An issue
link also works where the reasoning is a discussion rather than a description.

## Code

- `auto` parameters over a template type parameter used only as that parameter's type
- `namespace a::b {` at column 0; a more specific namespace over indented nested blocks
- early returns, small free functions over deep lambdas
- a branch per case with `requires`-constrained overloads, not `if constexpr` chains
- fewer template parameters and instantiations: fold a new axis into an existing one, or make it
  a runtime value
- complex parameter sets in a DAO: `f(Dao const&)` does the work, `f(first, auto&&... args)`
  builds the DAO from plain arguments
- members default initialized unless every constructor sets them; `using X_t = Outer::X;` for
  dependent types
- run-fixed layouts: an enum per variant (index = order), no scans by name
- device kernels capture plain data only (views, pointers, `std::array`); device data stays on
  the device, no host loops over unified memory
- no names that system headers define as macros (`B0`, `ECHO`, `VMIN`, ...)

## Memory and threads

Code that runs every step (per level, patch, tile or particle) does not allocate. No owning
container (`std::vector`, `std::string`, maps) created as a local in such a function: its
buffers are reused, from one of these, in order of preference:

- **owned by what is worked on**: a patch's temporary fields, a plan's scratch. Sized at setup
  or regrid, not per step. Work on a patch touches only that patch's storage, so threads on
  different patches never share a buffer
- **per thread**: `damn::core::scratch<T, Tag>(n)`, a `thread_local` buffer that only grows; its
  values are left over from earlier uses, and two uses alive at the same time on one thread take
  different tags. A `thread_local damn::MinimizingVector` where the size swings over the run: it
  gives memory back after being oversized for a while
- **fixed size batches**: a statically sized buffer (`std::array`), filled and flushed, where the
  total is unbounded or unknown up front

Setup and rare paths (input, decomposition, regrid planning, diagnostics) may allocate.

Threads:

- every buffer has one owner: an object (shared across threads only when read only) or a thread
  (`thread_local`). No function `static` written from several threads
- `thread_local` is per thread, not per call: code re-entered on the same thread (nested pools,
  recursion) does not reuse a buffer that is still in use
- no false sharing: values written by different threads at the same time sit on different cache
  lines. Accumulate per patch or tile, then reduce; per thread slots padded to
  `std::hardware_destructive_interference_size`
- the same holds on the device: device buffers are kept between launches (`thread_local
  gpu::Vec_t`), not allocated per launch

## Gates

- clang-format 22 on changed lines, iwyu (`iwyu_fix`), clang-tidy on changed lines, `-W 9`
- `res/ci/local.sh` before a PR is marked ready
