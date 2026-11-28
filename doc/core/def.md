# Build macros (`def.hpp`)

## DAO overloads: `DAMN_DAO_FN`, `DAMN_DAO_RET_FN`

A function taking a parameter set as a DAO, `fn(Dao const&)`, is also callable with the DAO's
members as plain arguments, `fn(args...)`. The macro writes that second overload: it builds
`Dao{args...}` and calls `fn` with it. Call sites then pass plain arguments and need not know the
DAO.

| Macro | Overload |
|---|---|
| `DAMN_DAO_FN(spec, fn, Dao)` | `spec auto fn(first, args...)`, `spec` its specifiers (e.g. `inline`, `DAMN_ALL`) |
| `DAMN_DAO_RET_FN(spec, fn, Dao)` | the same, `[[nodiscard]]`: `fn`'s result is the point of calling it. `fn` returning `void` is a compile error |

They follow mkn.kul's `MKN_KUL_FN_SIG` / `MKN_KUL_FN_SIG_RET`.

### Rules

- `Dao{args...}` is aggregate initialization: the members in order, missing trailing ones
  defaulted.
- `Dao` is a template name; its template arguments come from aggregate deduction (C++20, no
  deduction guides). Each template argument must appear in some member's type, or it cannot be
  deduced.
- `Dao` cannot be parenthesised in the macro: it is deduced from the braces that follow it.
- The overload is not used when the first argument already is a `Dao` (`Dao{dao}` would be a
  copy of the same type), so it never shadows `fn(Dao const&)`.
- A DAO used on the device holds views, not references.

## Inlining

`DAMN_ALWAYS_INLINE` and `DAMN_LAMBDA_ALWAYS_INLINE` are aliases of mkn.kul's
`MKN_KUL_ALWAYS_INLINE` (a function) and `MKN_KUL_LAMBDA_ALWAYS_INLINE` (after a lambda's
parameter list), until their users move to kul's. kul's `MKN_KUL_INLINE_CALL` inlines a call.
`[[nodiscard]]` is kul's `MKN_KUL_FN_SIG_RET`; damn has no macro of its own.

## Debug checks

| Macro | On when |
|---|---|
| `DAMN_DEBUG_DO(...)` | not `NDEBUG`, or `DAMN_FORCE_DEBUG_DO`: light checks |
| `DAMN_SUPER_DEBUG_DO(...)` | `-DDAMN_SUPER_DEBUG` (1; `=0` off), in any build, `-g 0` too: heavy checks (whole level sums, per stage invariants) |
| `DAMN_ASSERT(...)` | unless `DAMN_UNDEF_ASSERT`: CUDA can fail to compile with assertions |
