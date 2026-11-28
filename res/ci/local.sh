#!/usr/bin/env bash
# what CI runs (build_nix.yml, format.yml), locally: before marking a PR ready, instead of
# spending Actions minutes. From the repository root; JOBS=8 compile jobs by default.
#   res/ci/local.sh [base, default origin/next if it exists, else origin/master]
set -euo pipefail
BASE=${1:-$(git rev-parse -q --verify origin/next >/dev/null && echo origin/next || echo origin/master)}
J=${JOBS:-8}
mb=$(git merge-base "$BASE" HEAD)

step() { echo; echo "== $*"; }

# warnings as CI's: -W 9 (-Wall -Wextra -pedantic -Werror with gcc) except unused
#  parameters and variables, and strict aliasing a warning only (third party sources)
W=(-W 9 -a "-Wno-unused-parameter -Wno-unused-variable -Wno-error=strict-aliasing")

# every build output (dependencies' too) from this run: no library left from an older build, as a
#  .so an -lname would pick over the .a just built. ccache keeps the rebuild cheap
step "clean"
mkn clean -dp test_damn_core -x res/mkn/thrust
mkn clean -dp test_damn_hdf5 -x res/mkn/mpi

step "test executables (bin/<profile>/test): core, MPI, HDF5"
mkn build -p test_damn_core -dKO -t "$J" "${W[@]}" -x res/mkn/thrust
mkn build -p test_damn_mpi -dKO -t "$J" "${W[@]}" -x res/mkn/mpi
mkn build -p test_damn_hdf5 -dKO -t "$J" "${W[@]}" -x res/mkn/mpi

# as CI: tst/.phlop.exec.yaml, each test once, most MPI ones also on 2, 3 and 8 ranks (-c 8: an 8
#  rank job, oversubscribed, runs alone); phlop from tpp/phlop (the checkout mkn's sub: made)
step "tests (phlop)"
PYTHONPATH=tpp/phlop${PYTHONPATH:+:$PYTHONPATH} python3 -m phlop.run.test_cases -i tst -c 8 --tags mpi

step "clang-format (changed lines)"
# git clang-format --diff exits 1 on a diff: under pipefail a pipe into grep would not fail the
#  step, so the diff is captured, then looked at
fmt=$(git clang-format-22 --binary clang-format-22 --extensions c,cc,cpp,h,hpp,ipp --diff "$mb" -- inc src tst || true)
if grep -q "^diff" <<<"$fmt"; then
  echo "$fmt"
  echo "clang-format would change files: git clang-format-22 $mb"
  exit 1
fi

step "iwyu (applying its fixes changes nothing)"
mkn build -dKOqp iwyu_fix -t "$J" -Wx res/mkn/mpi >/dev/null
git diff --exit-code -- inc src tst

step "clang-tidy (changed lines)"
python3 res/ci/compile_db.py test_damn_core,test_damn_mpi,test_damn_hdf5 -dKO -x res/mkn/mpi
git diff -U0 "$mb" -- inc src tst \
  | clang-tidy-diff-22.py -clang-tidy-binary clang-tidy-22 -p1 -path . -j "$J" -quiet \
                          -iregex '.*\.(c|cc|cpp|h|hpp|ipp)' || true

echo; echo "== all done"
