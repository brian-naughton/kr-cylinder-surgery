#!/usr/bin/env python3
"""Replay comparison for a certificate ledger.

A runner is re-executed in a fresh process in a scratch directory, and the
ledger it writes there is compared against the frozen ledger shipped in
``runs_synthesis/``. Every field must agree except ``wall_time_s``, which is
the only clock-derived value any ledger is permitted to carry: nothing
derived from the clock may enter a check note, a section, or any other field,
precisely so that this comparison is exact everywhere else.

Usage:  replay_check.py FRESH_LEDGER FROZEN_LEDGER
Exits 0 if the two agree, 1 otherwise.
"""

from __future__ import annotations

import json
import sys


def strip_timings(obj: object) -> object:
    """Recursively drop ``wall_time_s`` entries from a decoded ledger."""
    if isinstance(obj, dict):
        return {k: strip_timings(v) for k, v in obj.items() if k != "wall_time_s"}
    if isinstance(obj, list):
        return [strip_timings(v) for v in obj]
    return obj


def load(path: str) -> object:
    with open(path) as fh:
        return strip_timings(json.load(fh))


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    fresh, frozen = argv[1], argv[2]
    try:
        if load(fresh) == load(frozen):
            return 0
    except (OSError, json.JSONDecodeError) as exc:
        print(f"REPLAY ERROR: {exc}")
        return 1
    print(f"REPLAY MISMATCH: {fresh} differs from {frozen}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
