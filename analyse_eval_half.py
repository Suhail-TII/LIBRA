#!/usr/bin/env python3
"""
analyse_eval_half.py -- the held-out result over all 103 configurations.

The eval-split run is complete: each open-weight target has a 35,020-row grid
(103 x 340). It was written several times over, because each dependent array
task resumed from the checkpoint and re-emitted the whole grid, so the job is
to pick ONE grid per target, prove the duplicates agree with it, and only then
compute anything.

What it reports, per target:

  probe c*        the configuration that wins on the 180 probe instructions
  eval ASR        what that configuration scores on the 340 held-out ones
  best on eval    the configuration that actually wins on the held-out 340
  forfeit         eval ASR of the best, minus eval ASR of the probe-selected one
                  -- the cost of having chosen on the probe half
  Spearman        rank correlation of all 103 configurations between halves

This supersedes the ten-configuration check in Section IV-C, which could not
establish the held-out optimum over the whole space.

    python analyse_eval_half.py
    python analyse_eval_half.py --root results/eval_half --probe results/replay_cache.parquet
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

N_CONFIG, N_INSTR = 103, 340
EXPECT = N_CONFIG * N_INSTR
OPEN_WEIGHT = ["Llama-2-7B-Chat", "Llama-3.1-8B-Instruct",
               "Qwen3-14B", "Ministral-3-8B-Instruct"]
KEY = ["config", "instruction_id"]
YCOL = "success_strict"


def full_grids(root: str) -> dict[str, list[str]]:
    """Per model, every file holding exactly one complete grid."""
    out: dict[str, list[str]] = {}
    for f in sorted(glob.glob(os.path.join(root, "*.csv"))):
        b = os.path.basename(f)
        if not (b.startswith("ckpt_") or b.startswith("eval_")):
            continue
        if os.path.getsize(f) < 20e6:          # the 3,400-row leftovers
            continue
        try:
            d = pd.read_csv(f, usecols=["model"])
        except Exception:                                   # noqa: BLE001
            continue
        ms = d.model.unique()
        if len(ms) != 1 or len(d) != EXPECT:
            continue
        out.setdefault(str(ms[0]), []).append(f)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/eval_half")
    ap.add_argument("--probe", default="results/replay_cache.parquet")
    ap.add_argument("--out", default="results/eval_half/heldout_103.csv")
    a = ap.parse_args()

    grids = full_grids(a.root)
    if not grids:
        print(f"no complete {EXPECT}-row grid found under {a.root}",
              file=sys.stderr)
        return 2

    # ---- probe half -------------------------------------------------------
    if not os.path.exists(a.probe):
        print(f"probe grid {a.probe} not found; pass --probe", file=sys.stderr)
        return 2
    pr = (pd.read_parquet(a.probe) if a.probe.endswith(".parquet")
          else pd.read_csv(a.probe))
    pcol = next((c for c in ("success", "success_strict", "y")
                 if c in pr.columns), None)
    if pcol is None:
        print(f"no success column in {a.probe}: {list(pr.columns)}",
              file=sys.stderr)
        return 2

    rows = []
    print(f"{'target':<26}{'grids':>6}  chosen file")
    chosen: dict[str, str] = {}
    for m in OPEN_WEIGHT:
        fs = grids.get(m, [])
        if not fs:
            print(f"{m:<26}{0:>6}  -- none")
            continue
        ck = [f for f in fs if os.path.basename(f).startswith("ckpt_")]
        pick = ck[0] if ck else sorted(fs)[-1]
        chosen[m] = pick
        print(f"{m:<26}{len(fs):>6}  {os.path.basename(pick)}")
    print()

    # ---- do the duplicate grids agree? ------------------------------------
    print("cross-checking the duplicate grids against the chosen one:")
    disagree_any = False
    for m, pick in chosen.items():
        base = pd.read_csv(pick, usecols=KEY + [YCOL]).set_index(KEY)[YCOL]
        for f in sorted(grids[m]):
            if f == pick:
                continue
            other = pd.read_csv(f, usecols=KEY + [YCOL]).set_index(KEY)[YCOL]
            j = base.to_frame("a").join(other.to_frame("b"), how="outer")
            nmiss = int(j.a.isna().sum() + j.b.isna().sum())
            ndiff = int((j.a.fillna(-1) != j.b.fillna(-1)).sum()) - nmiss
            flag = "ok" if (ndiff == 0 and nmiss == 0) else "DIFFERS"
            if flag != "ok":
                disagree_any = True
            print(f"  {m:<26}{os.path.basename(f)[:44]:<46}"
                  f"{ndiff:>6} diff {nmiss:>5} missing  {flag}")
    if disagree_any:
        print("\nThe duplicate grids do not agree. Greedy decoding should make "
              "re-runs\nidentical, so this needs explaining before any number "
              "goes in the paper.", file=sys.stderr)
        return 1
    print("  all duplicates identical\n")

    # ---- the held-out comparison ------------------------------------------
    for m, pick in chosen.items():
        ev = pd.read_csv(pick, usecols=["config", "instruction_id", YCOL])
        if ev.config.nunique() != N_CONFIG or ev.instruction_id.nunique() != N_INSTR:
            print(f"{m}: {ev.config.nunique()} configs x "
                  f"{ev.instruction_id.nunique()} instructions, expected "
                  f"{N_CONFIG} x {N_INSTR}", file=sys.stderr)
            return 1
        e = ev.groupby("config")[YCOL].mean() * 100
        p = (pr[pr.model == m].groupby("config")[pcol].mean() * 100)
        both = p.index.intersection(e.index)
        if len(both) != N_CONFIG:
            print(f"{m}: only {len(both)} configurations in common between "
                  f"halves", file=sys.stderr)
            return 1
        p, e = p[both], e[both]
        pstar = p.idxmax()
        estar = e.idxmax()
        rho = float(pd.Series(p).corr(pd.Series(e), method="spearman"))
        rows.append(dict(model=m, probe_cstar=pstar, probe_asr=float(p[pstar]),
                         eval_asr_of_probe_cstar=float(e[pstar]),
                         eval_best_config=estar, eval_best_asr=float(e[estar]),
                         forfeit=float(e[estar] - e[pstar]),
                         same=bool(pstar == estar), spearman=rho))

    r = pd.DataFrame(rows)
    print(f"{'target':<26}{'probe c* ASR':>13}{'on eval':>9}"
          f"{'best eval':>11}{'forfeit':>9}{'same?':>7}{'rho':>7}")
    for _, x in r.iterrows():
        print(f"{x.model:<26}{x.probe_asr:>12.1f}%{x.eval_asr_of_probe_cstar:>8.1f}%"
              f"{x.eval_best_asr:>10.1f}%{x.forfeit:>9.2f}"
              f"{str(x.same):>7}{x.spearman:>7.3f}")
    print(f"\n{'':<26}{'mean forfeit':>13} {r.forfeit.mean():.2f} points, "
          f"max {r.forfeit.max():.2f}; probe optimum also best on eval on "
          f"{int(r.same.sum())} of {len(r)} targets")
    print(f"{'':<26}{'Spearman':>13} {r.spearman.min():.2f} to "
          f"{r.spearman.max():.2f}")

    print("\nprobe-selected optimum per target:")
    for _, x in r.iterrows():
        print(f"  {x.model:<26}{x.probe_cstar}")
        if not x.same:
            print(f"  {'':<26}eval winner: {x.eval_best_config}")

    r.to_csv(a.out, index=False)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
