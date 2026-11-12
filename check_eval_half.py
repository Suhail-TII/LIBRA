#!/usr/bin/env python3
"""
check_eval_half.py -- is the held-out grid actually complete?

The eval-split run should give, for each of the four open-weight targets,
every one of the 103 configurations scored on all 340 evaluation instructions:

    103 x 340 = 35,020 rows per target,  140,080 rows in all.

Run this BEFORE run_eval_half.py --analyse. That script ranks configurations
by their mean success without weighting by how many instructions each was
actually scored on, so a target that is 90% finished still produces a full
set of plausible-looking numbers, with the unfinished configurations ranked
on whatever subset happens to exist. There is no error and nothing looks
wrong. This refuses to give a green light until the counts are exact.

    python check_eval_half.py
    python check_eval_half.py --root results/eval_half
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

N_CONFIG, N_INSTR = 103, 340
EXPECT = N_CONFIG * N_INSTR
OPEN_WEIGHT = ["Llama-2-7B-Chat", "Llama-3.1-8B-Instruct",
               "Qwen3-14B", "Ministral-3-8B-Instruct"]


def find(root: str) -> list[str]:
    pats = ["**/*.parquet", "**/*.csv"]
    out: list[str] = []
    for p in pats:
        out += glob.glob(os.path.join(root, p), recursive=True)
    # the pipeline's own loader skips these; so do we
    return sorted(f for f in out
                  if not any(k in os.path.basename(f).lower()
                             for k in ("profile", "runmeta", "manifest")))


def load(f: str) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
    except Exception as e:                                  # noqa: BLE001
        print(f"  !! {os.path.relpath(f)}: {type(e).__name__}: {e}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/eval_half")
    a = ap.parse_args()

    if not os.path.isdir(a.root):
        cands = [d for d in glob.glob("results/*") if os.path.isdir(d)]
        print(f"{a.root} not found. Directories under results/:", file=sys.stderr)
        for c in sorted(cands):
            print("   ", c, file=sys.stderr)
        print("\nRe-run with --root pointing at the eval-split outputs.",
              file=sys.stderr)
        return 2

    files = find(a.root)
    if not files:
        print(f"no parquet or csv under {a.root}", file=sys.stderr)
        return 2
    print(f"{len(files)} result file(s) under {a.root}\n")

    frames = [d for d in (load(f) for f in files) if d is not None]
    if not frames:
        return 2
    df = pd.concat(frames, ignore_index=True)

    need = {"model", "config", "instruction_id"}
    missing = need - set(df.columns)
    if missing:
        print(f"columns {sorted(missing)} absent; found {list(df.columns)}",
              file=sys.stderr)
        return 2

    ycol = next((c for c in ("success", "strict", "y", "label")
                 if c in df.columns), None)

    dup = int(df.duplicated(["model", "config", "instruction_id"]).sum())
    print(f"{'target':<26}{'rows':>9}{'of':>9}{'configs':>9}{'instr':>8}"
          f"{'%':>7}  status")
    short = []
    for m in OPEN_WEIGHT:
        d = df[df.model == m]
        n = len(d)
        nc, ni = d.config.nunique(), d.instruction_id.nunique()
        pct = 100.0 * n / EXPECT
        ok = (n == EXPECT and nc == N_CONFIG and ni == N_INSTR)
        if not ok:
            short.append(m)
        print(f"{m:<26}{n:>9,}{EXPECT:>9,}{nc:>9}{ni:>8}{pct:>6.1f}%  "
              f"{'complete' if ok else 'INCOMPLETE'}")

    others = sorted(set(df.model.unique()) - set(OPEN_WEIGHT))
    if others:
        print("\nalso present (not part of the held-out grid):")
        for m in others:
            print(f"  {m:<26}{len(df[df.model == m]):>9,} rows")

    print()
    if dup:
        print(f"  !! {dup:,} duplicate (model, config, instruction_id) rows: "
              f"a partial re-run was probably concatenated. Deduplicate "
              f"before analysing.")
    if ycol is None:
        print("  !! no success/label column found; this counts coverage only.")
    else:
        na = int(df[ycol].isna().sum())
        if na:
            print(f"  !! {na:,} rows have a null {ycol}.")

    if short or dup:
        print(f"\nNOT READY. Incomplete: {short or 'none'}."
              f"\nDo not run run_eval_half.py --analyse yet: it would rank "
              f"configurations\non unequal numbers of instructions and return "
              f"numbers that look fine.")
        return 1

    print(f"READY. All four targets at {EXPECT:,} rows "
          f"({4 * EXPECT:,} total).\n\nNow run:\n"
          f"  python run_eval_half.py --analyse\n\n"
          f"and send me the output plus this summary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
