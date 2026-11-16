#!/usr/bin/env python3
"""
audit_frame.py -- reconcile the human-audit frame against the planned grid.

    python audit_frame.py --inputs 'results/stage_b/*.csv' --out validation/frame.json

This mirrors how sample_validation.py builds its population, which is simply

    load every stage_b CSV, then keep the rows with a non-empty response.

It does not merge the decoded set and does not deduplicate, because neither
does the sampler: the decoded files only correct labels on rows already in the
frame. The planned grid for the final design is

    88 cells x (4 x 180 + 3 x 178) = 110,352 rows,

four open-weight targets at 180 probe instructions and three API-served
endpoints at 178. The frame can differ from that in two directions, and this
reports both per model: repeated keys, which resumed array jobs can produce,
add rows, while empty responses remove them.
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import pathlib
import sys

import pandas as pd

PLANNED_CELLS = 88
PROBE = {"Llama-2-7B-Chat": 180, "Llama-3.1-8B-Instruct": 180,
         "Qwen3-14B": 180, "Ministral-3-8B-Instruct": 180,
         "gpt-5.1": 178, "gpt-5-mini": 178, "gpt-5-nano": 178}
RESP = ["full_response", "response", "completion", "output", "generation", "text"]
CFG = ["config", "cell", "configuration", "config_id", "attack", "variant",
       "attack_id", "combo"]


def pick(df, names, what, required=True):
    for n in names:
        if n in df.columns:
            return n
    if required:
        sys.exit(f"no {what} column; looked for {names}\n"
                 f"columns present: {list(df.columns)}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=["results/stage_b/*.csv"])
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("validation/frame.json"))
    a = ap.parse_args()

    files = sorted({f for p in a.inputs for f in globmod.glob(p)})
    if not files:
        sys.exit(f"no files matched {a.inputs}")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files],
                   ignore_index=True)
    print(f"\n  read {len(files)} file(s), {len(df):,} rows")

    resp = pick(df, RESP, "response")
    cfg = pick(df, CFG, "configuration")
    key = ["model", cfg, "instruction_id"]
    missing_key = [c for c in key if c not in df.columns]
    if missing_key:
        sys.exit(f"missing key column(s) {missing_key}; present: {list(df.columns)}")
    print(f"  response column '{resp}', configuration column '{cfg}'")
    print(f"  key = {key}")

    nonempty = df[resp].notna() & (df[resp].astype(str).str.strip() != "")
    rows, detail = [], {}
    for model, g in df.groupby("model"):
        ge = g[nonempty.loc[g.index]]
        n_cells = int(g[cfg].nunique())
        n_instr = int(g["instruction_id"].nunique())
        rows.append({
            "model": model,
            "cells": n_cells, "instructions": n_instr,
            "structural": n_cells * n_instr,
            "raw": len(g), "empty": int(len(g) - len(ge)), "frame": len(ge),
            "repeated_keys": int(ge.duplicated(key).sum()),
            "cells_in_frame": int(ge[cfg].nunique()),
        })
        if len(g) - len(ge):
            by = (g[~nonempty.loc[g.index]].groupby(cfg).size()
                  .sort_values(ascending=False))
            detail[model] = {"cells_with_empties": int((by > 0).sum()),
                             "worst": {str(k): int(v) for k, v in by.head(5).items()},
                             "cells_fully_empty": [str(k) for k, v in by.items()
                                                   if v == n_instr]}

    tot = {k: sum(r[k] for r in rows) for k in
           ("structural", "raw", "empty", "frame", "repeated_keys")}
    w = "  {:<26}{:>7}{:>7}{:>11}{:>9}{:>8}{:>10}"
    print("\n" + w.format("model", "cells", "instr", "cells*instr",
                          "raw", "empty", "frame"))
    print("  " + "-" * 78)
    for r in sorted(rows, key=lambda x: x["model"]):
        flag = "" if r["cells_in_frame"] == r["cells"] else \
            f"   <- {r['cells'] - r['cells_in_frame']} cell(s) wholly empty"
        print(w.format(r["model"], r["cells"], r["instructions"],
                       f"{r['structural']:,}", f"{r['raw']:,}",
                       f"{r['empty']:,}", f"{r['frame']:,}") + flag)
    print("  " + "-" * 78)
    print(w.format("total", "", "", f"{tot['structural']:,}", f"{tot['raw']:,}",
                   f"{tot['empty']:,}", f"{tot['frame']:,}"))

    print(f"\n  raw {tot['raw']:,} minus empty {tot['empty']:,} = "
          f"frame {tot['frame']:,}"
          f"{'  OK' if tot['raw'] - tot['empty'] == tot['frame'] else '  MISMATCH'}")
    print(f"  repeated keys in the frame: {tot['repeated_keys']:,}")
    gaps = [r for r in rows if r["raw"] != r["structural"]]
    if gaps:
        print("\n  rows not equal to cells x instructions, so the grid is ragged:")
        for r in gaps:
            print(f"    {r['model']:<26}{r['raw']:,} against "
                  f"{r['structural']:,} ({r['raw'] - r['structural']:+,})")
    if detail:
        print("\n  empty responses by configuration:")
        for m, d in detail.items():
            print(f"    {m}: {d['cells_with_empties']} cell(s) affected; "
                  f"wholly empty: {d['cells_fully_empty'] or 'none'}")
            print(f"      largest: " + ", ".join(f"{k} {v}" for k, v in d["worst"].items()))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"response_column": resp, "configuration_column": cfg,
         "per_model": rows, "empty_detail": detail, "totals": tot}, indent=2) + "\n")
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
