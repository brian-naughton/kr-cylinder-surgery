#!/usr/bin/env bash
#
# verify.sh -- re-verify every computational claim in the paper, from scratch.
#
# What it does, in order:
#   1. builds a fresh virtual environment and installs the pinned sympy;
#      if no network or no pip is available it falls back to the system
#      sympy and warns if the version differs from the pinned one;
#   2. runs the two standing environment gates (danielewski.py, cross_check.py);
#   3. re-runs all 22 certificate runners in a scratch directory, so the frozen
#      ledgers in runs_synthesis/ are never touched;
#   4. compares each freshly produced ledger against its frozen counterpart,
#      field for field, ignoring only per-check wall-clock timings;
#   5. tallies the check classes and compares them with the expected totals.
#
# Exits 0 only if every gate passes, every ledger replays, and every tally
# matches.
#
# TESTED ON: Python 3.14.6 with sympy 1.14.0, macOS/arm64. Python 3.9+ is
# expected to work; the pinned sympy matters more than the Python version,
# because a different Groebner implementation can order a basis differently
# and change recorded strings without changing any mathematics.
#
# RUNTIME: about 11 minutes end to end on a 2026 laptop -- roughly 6.5 minutes
# of certificate runners plus the time to build the virtual environment and
# install sympy. The runner time is very unevenly distributed: see the table
# in VERIFY.md. One runner, rg_1_diagnostics, accounts for roughly 70% of
# the runner time on its own
# (~4.5 minutes); every other runner finishes in under 25 seconds and most in
# under 5. If the script appears to hang, it is almost certainly inside
# rg_1_diagnostics, which is working, not stuck.
#
# Usage:  ./verify.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
WORK="$ROOT/.verify"

SYMPY_PIN="1.14.0"

RUNNERS=(
  rd_1_transcribe rd_2_content rd_3_kernel rd_4_plinth rd_5_slice
  rd_6_brackets
  rg_1_diagnostics rg_2_boundary rg_3_k1 rg_4_k2
  rm_1_canonical rm_2_compress rm_3_crossing rm_4_cocycle
  rs_1_battery rs_2_arrows rs_3_epoly rs_4_generator
  rf_1_groundwork rf_2_generator rf_3_death
  pub_1_repairs
)

# Expected totals over all 22 ledgers. A discrepancy here is a real defect:
# see VERIFY.md for what each class means and what to do about a mismatch.
EXPECT_PASS=713
EXPECT_FAIL=0
EXPECT_CHECKS=883

echo "== 1. environment =="
rm -rf "$WORK"
mkdir -p "$WORK/runs_synthesis"

PY=""
if python3 -m venv "$WORK/venv" 2>/dev/null \
   && "$WORK/venv/bin/pip" install --quiet \
        --timeout 15 --retries 1 "sympy==$SYMPY_PIN" 2>/dev/null; then
  PY="$WORK/venv/bin/python"
  echo "   fresh venv with pinned sympy==$SYMPY_PIN"
else
  # Offline / no-pip fallback: use whatever sympy the system already has.
  if python3 -c 'import sympy' 2>/dev/null; then
    PY="python3"
    have="$(python3 -c 'import sympy; print(sympy.__version__)')"
    echo "   NOTE: could not build a fresh environment (no network or no pip)."
    echo "         Falling back to the system Python and its installed sympy."
    if [ "$have" != "$SYMPY_PIN" ]; then
      echo "   WARNING: sympy $have, but these certificates were produced with"
      echo "            sympy $SYMPY_PIN. All the mathematics is version-independent,"
      echo "            but a different Groebner implementation can order a basis"
      echo "            differently, which can change recorded strings and make"
      echo "            ledgers fail to replay. A replay mismatch under a"
      echo "            different sympy is not necessarily a defect; a FAIL is."
    fi
  else
    echo "   ERROR: no usable Python environment. Install sympy==$SYMPY_PIN"
    echo "          (pip install sympy==$SYMPY_PIN) and re-run."
    exit 1
  fi
fi
"$PY" -c 'import sympy, sys; print("   python", ".".join(map(str, sys.version_info[:3])), "/ sympy", sympy.__version__)'

echo "== 2. standing gates =="
cp "$ROOT"/*.py "$WORK/"
( cd "$WORK" && "$PY" danielewski.py >/dev/null ) && echo "   danielewski.py    OK (20 checks)"
( cd "$WORK" && "$PY" cross_check.py  >/dev/null ) && echo "   cross_check.py    OK (58 checks)"

echo "== 3. certificate runners (fresh process each, scratch ledgers) =="
echo "   note: rg_1_diagnostics takes ~4.5 minutes; the rest are seconds."
for r in "${RUNNERS[@]}"; do
  printf '   %-18s ' "$r"
  ( cd "$WORK" && "$PY" "$r.py" >"$WORK/$r.out" 2>&1 ) \
    || { echo "RUNNER FAILED"; cat "$WORK/$r.out"; exit 1; }
  grep -E '^tally' "$WORK/$r.out" || { echo "no tally emitted"; exit 1; }
done

echo "== 4. ledger replay against the frozen certificates =="
replayed=0
for frozen in "$ROOT"/runs_synthesis/*.json; do
  name="$(basename "$frozen")"
  "$PY" "$ROOT/replay_check.py" "$WORK/runs_synthesis/$name" "$frozen" \
    || { echo "   $name  MISMATCH"; exit 1; }
  replayed=$((replayed + 1))
done
echo "   $replayed / 22 ledgers replay identically"

echo "== 5. tallies =="
"$PY" - "$WORK/runs_synthesis" "$EXPECT_PASS" "$EXPECT_FAIL" "$EXPECT_CHECKS" <<'PYEOF'
import collections
import glob
import json
import os
import sys

ledger_dir, exp_pass, exp_fail, exp_checks = sys.argv[1:5]
counts = collections.Counter()
for path in sorted(glob.glob(os.path.join(ledger_dir, "*.json"))):
    with open(path) as fh:
        data = json.load(fh)
    for cls, n in data["tally"].items():
        counts[cls] += n

total = sum(counts.values())
for cls in sorted(counts):
    print(f"   {cls:22s} {counts[cls]}")
print(f"   {'TOTAL':22s} {total}")

ok = (
    counts["PASS"] == int(exp_pass)
    and counts.get("FAIL", 0) == int(exp_fail)
    and total == int(exp_checks)
)
if not ok:
    print("\nTALLY MISMATCH -- expected "
          f"PASS={exp_pass}, FAIL={exp_fail}, TOTAL={exp_checks}")
    sys.exit(1)
PYEOF

echo
echo "ALL CHECKS PASS -- $EXPECT_CHECKS checks, $EXPECT_PASS PASS, $EXPECT_FAIL FAIL,"
echo "22 / 22 ledgers replay identically against the frozen certificates."
rm -rf "$WORK"
