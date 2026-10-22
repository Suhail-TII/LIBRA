#!/usr/bin/env python3
"""
run_eval_half.py — confirm that configurations selected on the probe half hold
up on the held-out evaluation half.

WHY THIS EXISTS
---------------
Section III freezes a 180/340 probe/evaluation split and states that every
design decision is made on probe, precisely so that reporting is not done on
the instructions used to choose. But the paper as drafted reports c*, ASR and
regret entirely on probe. The framings were retained as the union of each
target's top-3 on probe, and c* is then computed over those same instructions,
so the reported optima carry a selection bias of unknown size.

Measuring all 103 configurations on 340 evaluation instructions would cost
175,100 generations, more than the whole study. It is not necessary. The claim
that needs support is narrower: that a configuration chosen on probe is still
good on instructions that played no part in choosing it. Measuring each
target's top-k probe configurations on the evaluation half tests exactly that,
at k=10 for 17,000 generations.

WHAT IT ANSWERS
---------------
  1. Does c*(probe) remain the best, or near-best, on eval?
  2. How much does its ASR drop? That drop is the selection bias.
  3. Does the probe ranking of the top-k survive on eval (Spearman)?

If the ranking holds and the drop is small, the selection generalises and the
reported numbers stand with a stated caveat. If c*(probe) falls sharply on
eval, the paper must report eval ASR instead.

    python run_eval_half.py --plan          # what would run, no generation
    sbatch slurm/eval_half.sh               # open-weight targets
    python run_eval_half.py --analyse       # once results exist
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

POOL = ["Llama-2-7B-Chat", "Llama-3.1-8B-Instruct", "Qwen3-14B",
        "gpt-5.1", "Ministral-3-8B-Instruct"]
CACHE = "results/replay_cache.parquet"
SPLIT = "config/advbench_split_v1.json"
OUT = Path("results/eval_half")


def top_k_per_target(k: int) -> dict[str, list[str]]:
    """
    The k highest-scoring configurations per target on the probe half.

    Includes c* by construction. Taking the top k rather than c* alone lets us
    ask whether the probe RANKING survives, not merely whether the winner does:
    a procedure that picks any of the top few is doing its job, so rank
    stability matters more than the identity of the single best.
    """
    d = pd.read_parquet(CACHE)
    d = d[d.model.isin(POOL)]
    a = d.groupby(["model", "config"]).success.mean()
    return {m: a.loc[m].sort_values(ascending=False).head(k).index.tolist()
            for m in POOL}


def plan(k: int):
    sel = top_k_per_target(k)
    split = json.load(open(SPLIT))
    n_eval = len(split["eval"])
    print(f"evaluation instructions: {n_eval}")
    total = 0
    for m, cfgs in sel.items():
        n = len(cfgs) * n_eval
        total += n
        kind = "framing-only" if "|" not in cfgs[0] else "composed"
        print(f"\n{m}")
        print(f"  c*(probe): {cfgs[0]}  [{kind}]")
        print(f"  {len(cfgs)} configurations x {n_eval} instructions = {n:,}")
    print(f"\nTOTAL: {total:,} generations")
    print("Note: API targets are measured on 338 of the 340 evaluation "
          "instructions if the two filtered instructions fall in that half.")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "selection.json", "w") as fh:
        json.dump({"k": k, "n_eval": n_eval, "selection": sel}, fh, indent=2)
    print(f"\nwrote {OUT}/selection.json — the runners read this")


def analyse():
    """Compare probe and evaluation performance for the selected set."""
    sel = json.load(open(OUT / "selection.json"))["selection"]
    probe = pd.read_parquet(CACHE)
    probe = probe[probe.model.isin(POOL)]
    p = probe.groupby(["model", "config"]).success.mean()

    # Read the released cache rather than globbing the raw CSVs, so this
    # reproduces from the artifact alone.
    cache = Path("results/eval_half_cache.parquet")
    if cache.exists():
        e = pd.read_parquet(cache)
    else:
        files = [f for f in glob.glob(str(OUT / "eval_*.csv"))
                 if "profile" not in f and "runmeta" not in f]
        if not files:
            raise SystemExit(f"no {cache} and no results in {OUT}/")
        e = pd.concat([pd.read_csv(f, low_memory=False) for f in files],
                      ignore_index=True)
        e = e[~e.get("api_error", pd.Series(False, index=e.index)).fillna(False)]
    key = "cell" if "cell" in e.columns else "config"
    ev = e.groupby(["model", key]).success_strict.mean()

    print("=" * 78)
    print("PROBE vs HELD-OUT EVALUATION")
    print("=" * 78)
    rows = []
    for m in POOL:
        cfgs = [c for c in sel[m] if (m, c) in ev.index]
        if not cfgs:
            print(f"{m}: no evaluation data yet")
            continue
        pv = np.array([p.loc[m, c] for c in cfgs])
        evv = np.array([ev.loc[m, c] for c in cfgs])

        star_p = cfgs[int(np.argmax(pv))]      # best on probe
        star_e = cfgs[int(np.argmax(evv))]     # best on eval, within this set
        rho = pd.Series(pv).corr(pd.Series(evv), method="spearman")

        # The quantity that matters: how much the probe-selected winner loses
        # on instructions that played no part in selecting it.
        drop = p.loc[m, star_p] - ev.loc[m, star_p]
        # And what it forfeits against the best of the set on eval.
        forfeit = ev.loc[m, star_e] - ev.loc[m, star_p]

        rows.append(dict(model=m, probe=p.loc[m, star_p],
                         evalu=ev.loc[m, star_p], drop=drop,
                         best_eval=ev.loc[m, star_e], forfeit=forfeit,
                         rho=rho, same=star_p == star_e))
        print(f"\n{m}")
        print(f"  c*(probe)          {star_p}")
        print(f"    probe ASR        {p.loc[m, star_p]:.1%}")
        print(f"    eval  ASR        {ev.loc[m, star_p]:.1%}   "
              f"(drop {drop*100:+.1f} points)")
        print(f"  best on eval       {star_e}")
        print(f"    eval  ASR        {ev.loc[m, star_e]:.1%}")
        print(f"  probe-selected forfeits {forfeit*100:.1f} points on eval")
        print(f"  rank correlation over the {len(cfgs)} selected: {rho:+.2f}")

    if rows:
        r = pd.DataFrame(rows)
        print("\n" + "=" * 78)
        print(f"c*(probe) is also best on eval for {r.same.sum()}/{len(r)} targets")
        print(f"mean ASR drop probe -> eval : {r['drop'].mean()*100:+.1f} points")
        print(f"mean forfeit vs best on eval: {r['forfeit'].mean()*100:.1f} points")
        print(f"mean rank correlation       : {r['rho'].mean():+.2f}")
        r.to_csv(OUT / "probe_vs_eval.csv", index=False)
        print(f"\nwrote {OUT}/probe_vs_eval.csv")
        print("\nThe forfeit column is the number the paper needs: it is what a "
              "probe-selected\nconfiguration gives up on instructions that "
              "played no part in selecting it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()
    if args.analyse:
        analyse()
    else:
        plan(args.k)


if __name__ == "__main__":
    main()
