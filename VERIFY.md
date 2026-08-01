# Verifying the certificates

Everything computational in the paper is checked by machine, in exact
arithmetic over ℚ, and the checks are shipped with their results.

## One command

```bash
./verify.sh
```

It (1) builds a fresh virtual environment and installs `sympy`; (2) runs the
two standing environment gates; (3) re-runs all 22 certificate runners, each
in a fresh process, in a scratch directory `.verify/` so the frozen ledgers
are never touched; (4) replays each freshly produced ledger against its frozen
counterpart; (5) checks the tallies. It exits 0 only if all of that holds.

**Requirements.** Python 3.9+ and, on first run, network access to install the
pinned `sympy==1.14.0`. If neither pip nor the network is available, the
script falls back to the system `sympy` and warns when the version differs
from the pinned one — the mathematics is version-independent, but a different
Gröbner implementation can order a basis differently, which changes recorded
strings and can make a ledger fail to replay. Tested on **Python 3.14.6 with
sympy 1.14.0, macOS/arm64**.

**Runtime, measured rather than estimated.** About **11 minutes end to end**:
roughly **6.5 minutes** of certificate runners, plus the time to build the
virtual environment and install sympy. The runner time is very unevenly
distributed — one runner dominates:

| runner | wall time | | runner | wall time |
|---|---:|---|---|---:|
| `rg_1_diagnostics` | **~275 s** | | `rg_3_k1` | 7 s |
| `rm_4_cocycle` | 22 s | | `rd_4_plinth` | 5 s |
| `rd_1_transcribe` | 15 s | | `rm_2_compress` | 5 s |
| `rg_4_k2` | 12 s | | `rg_2_boundary` | 4 s |
| `rd_5_slice` | 3 s | | each `rs_*`, `rf_*`, `pub_1_repairs`, `rm_1`, `rm_3` | ~4 s |
| `rd_2_content` | 2 s | | `rd_3_kernel`, `rd_6_brackets` | <1 s |

`rg_1_diagnostics` alone is roughly **70%** of the wall time: it runs the
declared-cap plinth search (126 monomials, a 20-dimensional solution space)
together with the fibre-dimension computations behind the quotient-morphism
pathology. If the script appears to hang, it is almost certainly inside that
runner, and it is working rather than stuck. No runner approaches its declared
resource ceiling.

## Expected totals

| | |
|---|---|
| checks | **883** |
| PASS | **713** |
| FAIL | **0** |
| ledgers replaying identically | **22 / 22** |

plus the two standing gates, which are run first and are not part of those
totals: `danielewski.py` (20 checks) and `cross_check.py` (58 checks, of which
29 are twin-agreement checks).

Full breakdown by class:

| class | count | meaning |
|---|---|---|
| PASS | 713 | a certified identity or predicate |
| THEOREM | 15 | a statement proved exactly, with no degree bound |
| LEMMA | 10 | a topological or structural step taken over certified algebraic inputs |
| LEMMA-INPUT | 3 | an algebraic fact recorded for use by a lemma |
| MEASURED | 125 | the outcome of a **bounded search at a declared cap** — a measurement of what was searched, never a negative result |
| NOT-DETERMINED | 7 | asked, not settled, and recorded as such rather than guessed |
| NOT-ATTEMPTED | 2 | out of declared scope, or attempted once under an explicit bound and recorded as not landing |
| CANDIDATE-STATEMENT | 2 | a reading the evidence suggests but does not establish; asserted of nothing |
| CORRECTION | 1 | a prior expectation of ours overturned by a check |
| BOUNDARY | 2 | a limitation certified rather than glossed |
| ERRATUM | 2 | a slip in the published source paper (see the README) |
| REFUTED | 1 | a conjecture of ours, disproved by a check |
| FAIL | 0 | — |

The classes other than PASS/FAIL are the point of the exercise as much as the
passes are: a bounded search that finds nothing is recorded as MEASURED and
not as a proof of absence, and an open question is recorded as
NOT-DETERMINED and not quietly rounded to an answer.

## Reading a ledger

Each runner writes one JSON ledger to `runs_synthesis/`. The shipped copies
are read-only. Structure:

```
block          the block identifier, e.g. "RF-2"
date           the date the certificate was produced
source         attribution: whose objects are being computed with
plan           the internal working note that scoped the block (not shipped;
               see "Internal references" below)
entries[]      one per check: {check, outcome, wall_time_s}
sections{}     the structured findings, keyed by topic
notes{}        prose attached to individual findings
tally{}        counts per outcome class
total_checks   the number of entries
exit           the runner's exit status
```

**Provenance note.** Two ledgers — `RM-1-CANONICAL.json` and `RF-3-DEATH.json`
— were regenerated before release with corrected interpretive annotations, so
that their prose matches the paper; the computational payloads are unchanged,
and a check-for-check certification that only the enumerated prose fields
differ is held with the working record.

`wall_time_s` is the **only** clock-derived value any ledger carries. Nothing
derived from the clock may enter a note, a section, or any other field —
which is what makes the replay comparison in `replay_check.py` exact
everywhere else. A ledger that does not replay is a defect, not noise.

## What a mismatch means

- **A runner fails outright.** Almost always an environment problem — a
  `sympy` version whose Gröbner engine orders a basis differently. The
  mathematical content does not depend on that, but the ledger will differ;
  compare the failing entries, not the whole file.
- **A ledger mismatch with all checks passing.** The certificate is
  reproducing but some recorded string or structured value differs. Worth
  reporting.
- **A tally mismatch.** A real defect. Please open an issue.
- **A FAIL.** A real defect, and the more interesting kind. Please open an
  issue with the ledger.

## The runners

| runner | what it certifies | paper |
|---|---|---|
| `rd_1_transcribe.py` | the published starting datum, against its own printed identities | §2 |
| `rd_2_content.py` | the content of ∂ is exactly t³ | §3.1 |
| `rd_3_kernel.py` | ker ∂ = ℂ[p,q,t] | §3.1 |
| `rd_4_plinth.py` | degeneracy locus, plinth elements, the localisation theorem | §§3.2–3.4 |
| `rd_5_slice.py` | no slice, two ways, with a positive control | §3.2 |
| `rd_6_brackets.py` | all six brackets; the commuting triple | §3.5 |
| `rg_1_diagnostics.py` | plinth non-principality; the quotient-morphism pathology | §3.4 |
| `rg_2_boundary.py` | pole orders, the four relations, the tower | §§4.1, 5.1 |
| `rg_3_k1.py` | the two exact theorems at k = 1; the k = 1 sweep | §3.5 |
| `rg_4_k2.py` | 1-stability; the k = 2 sweep | §3.5 |
| `rm_1_canonical.py` | canonical centres; X₁ × 𝔸ᵏ ≇ 𝔸³⁺ᵏ; the walls | §§4.2, 5.1 |
| `rm_2_compress.py` | the compressed modification; the rectification obstruction | §§4.2–4.3 |
| `rm_3_crossing.py` | the square commutes; Y₁ smooth; the crossing | §5.2 |
| `rm_4_cocycle.py` | the transition datum; three bounded sweeps | §5.2 |
| `rs_1_battery.py` | the invariant battery; the three kills | §§6.1, 7.1 |
| `rs_2_arrows.py` | the four-arrow ledger; where the packet enters and dies | §7.2 |
| `rs_3_epoly.py` | the Hodge–Deligne collapse | §7.3 |
| `rs_4_generator.py` | the Euler-level mechanism and the defect point | §7.4 |
| `rf_1_groundwork.py` | σ₄ not proper; flatness; the doubled fibre; W ≃ S³ | §8.1 |
| `rf_2_generator.py` | H^BM₄(Y₁;ℤ) ≅ ℤ and the generator | §8.2 |
| `rf_3_death.py` | the defect cycle and its inverse image; the certified boundary | §§8.3–8.4 |
| `pub_1_repairs.py` | factoriality of A[w] (Nagata); smoothness of π; one bounded attempt at a local μ-certificate | §§3.1, 8.1 |

The last of these was added in the revision round after external review, and
it records one check that **did not land**: a bounded attempt at a local
certificate that the compressed centre is not a complete intersection. It is
kept in the ledger, marked NOT-ATTEMPTED with the reason, and the sentence it
would have supported was removed from the paper rather than weakened. The
rectification obstruction does not depend on it.

`rd_common.py`, `rg_common.py`, `rm_common.py`, `rs_common.py` and
`rf_common.py` hold the shared certified objects. Each later visit's runners
re-verify the earlier visits' facts **in process** before doing anything else,
so nothing is carried between sessions on trust.

`checker.py` and `checker_twin.py` are two independent ideal-membership
implementations; `cross_check.py` cross-validates them against each other, and
`danielewski.py` certifies a known isomorphism as a standing end-to-end gate
on the environment. Both are run first by `verify.sh`.

## Internal references

The ledgers and runner docstrings carry short internal codes from the working
protocol under which they were produced — block names (`RD-2`, `RF-1`), plan
files (`RD-OPS.md`), amendment numbers (`A22`, `A30`), hazard codes (`H15` for
a boundary counter-check, `H17` for the per-runner replay rule), and routing
notes. Those working documents are not part of this repository, and none of
the mathematics depends on them. They are left in place rather than scrubbed
because rewriting a frozen certificate to look tidier is exactly the thing the
freezing is meant to prevent: what is shipped is what was run.
