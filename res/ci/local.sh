#!/usr/bin/env bash
# what CI runs (build_nix.yml, format.yml), locally: before marking a PR ready, instead of
# spending Actions minutes. From the repository root; JOBS=8 compile jobs by default.
#   res/ci/local.sh [base, default origin/next: PRs target next]
set -euo pipefail
BASE=${1:-origin/next}
J=${JOBS:-8}
MPI="mpirun --oversubscribe"
mb=$(git merge-base "$BASE" HEAD)

step() { echo; echo "== $*"; }

# warnings as CI's: -W 9 (-Wall -Wextra -pedantic -Werror with gcc), PHARE's devMode exceptions
#  (unused parameters, variables), and strict aliasing a warning only (third party sources)
W=(-W 9 -a "-Wno-unused-parameter -Wno-unused-variable -Wno-error=strict-aliasing")

step "core tests"
mkn build test -p test_damn_core -dKO -t "$J" "${W[@]}" -x res/mkn/thrust

tests() { find "bin/$1/test" -type f -perm -u+x -name "*.cpp" | sort; }

step "MPI tests: 1 (mkn's run), 2, 3 and 8 ranks (as CI: 8 leaves ranks without a patch)"
mkn build test -p test_damn_mpi -dKO -t "$J" "${W[@]}" -x res/mkn/mpi
for n in 2 3 8; do
  for t in $(tests test_damn_mpi); do $MPI -n $n "$t" --gtest_brief=1; done
done

step "HDF5 tests: 1 and 3 ranks"
mkn build -p test_damn_hdf5 -dKO -t "$J" "${W[@]}" -x res/mkn/mpi
mkn build test -p test_damn_hdf5 -dKO -t "$J" "${W[@]}" -x res/mkn/mpi >/dev/null # links them
for n in 1 3; do
  for t in $(tests test_damn_hdf5); do (cd "$(mktemp -d)" && $MPI -n $n "$OLDPWD/$t" --gtest_brief=1); done
done

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
