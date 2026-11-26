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

## Gates

- clang-format 22 on changed lines, iwyu (`iwyu_fix`), clang-tidy on changed lines, `-W 9`
- `res/ci/local.sh` before a PR is marked ready
