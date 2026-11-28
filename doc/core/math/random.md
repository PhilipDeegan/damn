# Per cell random streams (`math/random.hpp`)

Seeded particle loading must not depend on how a level is cut into patches or spread over MPI
ranks. So a cell's random numbers come from what identifies the cell, never from the patch.

```
seed ──┐
pop  ──┼─ hash_combine chain ─> cell key ─> SplitMix64 ─> deltas, velocities of the cell
level ─┤
cell ──┘
```

`cell_key` hashes (user seed, population, level, global AMR cell index) into a 64 bit key, and
that key seeds a small generator used for that cell only.

## What goes into the key

| Input | Why |
|---|---|
| seed | the user's: another seed, other particles |
| population | two populations with the same seed must not get the same particles |
| level | the same AMR index exists on every level, but is a different physical cell |
| cell | the global AMR index, the same whichever patch or rank holds the cell |

Negative indices are sign extended to 64 bits, so -2 and 2 give different keys.

The combining order is seed, population, level, `cell[0]`, `cell[1]`, `cell[2]`. For a fixed
prefix the last combine is bijective, so cells differing in their last index never share a key;
other collisions have probability ~2^-64.

## Reproducibility

Everything is plain 64 bit unsigned integer arithmetic: unsigned overflow wraps modulo 2^64 by
definition in C++, so the keys are the same bits on every platform and compiler.
