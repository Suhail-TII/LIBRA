#!/usr/bin/env python3
"""
inspect_eval_half.py -- what are those 23 files?

check_eval_half.py concatenated everything it found and got 3 to 5 times the
expected row count with no success column, which means the glob is wrong, not
the run. Every target came out as an exact multiple of 35,020 plus 3,401,
which looks like several files each holding a full grid (shards, stages, or
the same data in two formats) rather than a partially finished run.

This prints one line per file with its shape, its columns and which models it
holds, and groups the files by column signature. It reads nothing into a
combined frame and changes nothing on disk.

    python inspect_eval_half.py
    python inspect_eval_half.py --root results/eval_half --peek
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys

import pandas as pd

EXPECT = 103 * 340


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/eval_half")
    ap.add_argument("--peek", action="store_true",
                    help="also print the first row of each column signature")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.root, "**", "*"), recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print(f"nothing under {a.root}", file=sys.stderr)
        return 2

    print(f"{len(files)} file(s) under {a.root}\n")
    print(f"{'file':<46}{'MB':>7}{'rows':>9}{'cols':>6}  models")
    sigs: dict[tuple, list[str]] = {}
    for f in files:
        rel = os.path.relpath(f, a.root)
        mb = os.path.getsize(f) / 1e6
        ext = os.path.splitext(f)[1].lower()
        if ext not in (".parquet", ".csv", ".tsv", ".json", ".jsonl"):
            print(f"{rel[:45]:<46}{mb:>7.1f}{'-':>9}{'-':>6}  (skipped: {ext})")
            continue
        try:
            if ext == ".parquet":
                d = pd.read_parquet(f)
            elif ext in (".csv", ".tsv"):
                d = pd.read_csv(f, sep="\t" if ext == ".tsv" else ",")
            else:
                d = pd.read_json(f, lines=(ext == ".jsonl"))
        except Exception as e:                              # noqa: BLE001
            print(f"{rel[:45]:<46}{mb:>7.1f}  !! {type(e).__name__}: {e}")
            continue
        mcol = next((c for c in ("model", "target", "model_name")
                     if c in d.columns), None)
        models = (",".join(sorted(str(x) for x in d[mcol].unique())[:3])
                  if mcol else "(no model column)")
        print(f"{rel[:45]:<46}{mb:>7.1f}{len(d):>9,}{len(d.columns):>6}  "
              f"{models[:46]}")
        sigs.setdefault(tuple(d.columns), []).append(rel)

    print(f"\n{len(sigs)} distinct column signature(s):\n")
    for i, (cols, fs) in enumerate(sigs.items(), 1):
        print(f"  [{i}] {len(fs)} file(s): {', '.join(f[:34] for f in fs[:4])}"
              f"{' ...' if len(fs) > 4 else ''}")
        print(f"      columns ({len(cols)}): {', '.join(map(str, cols))}")
        looks_judged = any(k in " ".join(map(str, cols)).lower() for k in
                           ("success", "strict", "harmbench", "llama_guard",
                            "lg_", "hb_", "label", "verdict", "judge"))
        print(f"      judged? {'yes' if looks_judged else 'NO -- generations only'}")
        if a.peek:
            f0 = os.path.join(a.root, fs[0])
            try:
                d = (pd.read_parquet(f0) if f0.endswith(".parquet")
                     else pd.read_csv(f0))
                print("      first row:")
                for k, v in d.iloc[0].items():
                    s = str(v).replace("\n", " ")
                    print(f"        {str(k):<22} {s[:70]}")
            except Exception:                               # noqa: BLE001
                pass
        print()

    print(f"One complete held-out grid is {EXPECT:,} rows per target "
          f"(103 x 340).\nPick the ONE signature that is judged and whose "
          f"files partition the grid,\nand tell me which it is -- I will "
          f"point the check at that set only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
