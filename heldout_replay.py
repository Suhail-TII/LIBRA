#!/usr/bin/env python3
"""
heldout_replay.py -- run LIBRA-S and all five baselines on the HELD-OUT grid.

Every regret figure in Section V is replayed over the same 180 probe
instructions that reduced the component pools, built the screen and defined
each target's optimum. This runs the identical procedure over the 340
evaluation instructions instead, scoring regret against the held-out optimum.
No new target queries: the 103 x 340 grid is already measured.

    python heldout_replay.py --check                 # gate only
    python heldout_replay.py --seeds 200 --matched

THE GATE. Before producing any held-out number, this replays the PROBE cache
and checks every method against results/per_seed_regrets.csv, the published
per-seed file behind Tables 5 and 6, at two budgets and on all five targets.
It refuses to continue on any mismatch. That matters because the baselines are
easy to get wrong: libra_select.bl_race_only is 'race_only_legacy' in that
file and is NOT the baseline the paper reports. Section V's full-space halving
is full_space_halving.bl_successive_halving, which races the whole space and
reallocates the remaining budget across the remaining rounds.

Reads:
    results/replay_cache.parquet      probe cache, for the gate
    results/per_seed_regrets.csv      published per-seed regrets, for the gate
    results/eval_half/*.csv           ckpt_/eval_ grids, 103 x 340
Writes:
    results/heldout_replay/heldout_replay.json

Scope to state in the paper: four open-weight targets, since the held-out grid
was never run on the API-served target; and the screen and the component pools
were still fixed on probe data, so this tests the search on fresh instructions
inside a space chosen beforehand.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

N_CONFIG, N_INSTR = 103, 340
EXPECT = N_CONFIG * N_INSTR
OPEN_WEIGHT = ["Llama-2-7B-Chat", "Llama-3.1-8B-Instruct",
               "Qwen3-14B", "Ministral-3-8B-Instruct"]
YCOL, KEY = "success_strict", ["config", "instruction_id"]
TOL = 0.01          # points, against the published per-seed means

try:
    from libra_select import (Replay, Hyper, select, CACHE, POOL,   # noqa: F401
                              bl_random, bl_transfer)
    from full_space_halving import bl_successive_halving
    from reviewer_analyses import bl_singles_sweep, bl_ucb
except Exception as e:                                     # noqa: BLE001
    sys.exit(f"cannot import the harness: {e}\n"
             "Run this from the repository root, in the environment that runs "
             "reviewer_analyses.py.")

# Paper names, matching the method column of results/per_seed_regrets.csv.
METHODS = {"LIBRA": select,
           "full_space_halving": bl_successive_halving,
           "random": bl_random,
           "singles_sweep": bl_singles_sweep,
           "ucb": bl_ucb,
           "transfer": bl_transfer}
RACE = "full_space_halving"                 # the baseline to beat
ANCHORED = [m for m in METHODS if m != "transfer"]   # transfer spends no queries


# ---------------------------------------------------------------- held-out IO

def full_grids(root):
    """Per model, every file holding exactly one complete 103 x 340 grid."""
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "*.csv"))):
        b = os.path.basename(f)
        if not (b.startswith("ckpt_") or b.startswith("eval_")):
            continue
        if os.path.getsize(f) < 20e6:
            continue
        try:
            d = pd.read_csv(f, usecols=["model"])
        except Exception:                                  # noqa: BLE001
            continue
        ms = d.model.unique()
        if len(ms) == 1 and len(d) == EXPECT:
            out.setdefault(str(ms[0]), []).append(f)
    return out


def load_heldout(root):
    grids = full_grids(root)
    missing = [m for m in OPEN_WEIGHT if m not in grids]
    if missing:
        sys.exit("no complete held-out grid for: " + ", ".join(missing) +
                 f"\nRun check_eval_half.py --root {root} first.")
    frames = []
    for m in OPEN_WEIGHT:
        pick = grids[m][0]
        d = pd.read_csv(pick, usecols=KEY + [YCOL])
        if len(d) != EXPECT:
            sys.exit(f"{pick}: {len(d)} rows, expected {EXPECT}")
        if d.duplicated(KEY).any():
            sys.exit(f"{pick}: duplicate (config, instruction_id) rows")
        d = d.rename(columns={YCOL: "success"})
        d["success"] = d["success"].astype(bool)
        d["model"] = m
        frames.append(d[["model", "config", "instruction_id", "success"]])
        extra = f"   ({len(grids[m])} complete grids; using the first)" \
            if len(grids[m]) > 1 else ""
        print(f"  {m:<26}{os.path.basename(pick):<34}"
              f"{d.success.mean():.1%} mean ASR{extra}")
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ the replay

def run(df, models, budgets, seeds, hp, label, methods=None, quiet=False):
    """Mean regret per (method, budget, model), and per-seed target means."""
    methods = methods or METHODS
    rows, blocks = [], {}
    for m in models:
        rp = Replay(df, m)
        best_c, best = rp.optimum()
        peers = [Replay(df, o).optimum()[0] for o in models if o != m]
        if not quiet:
            print(f"  {label}: {m:<26}c* = {best_c} at {best:.1%}")
        for B in budgets:
            for name, fn in methods.items():
                reg = []
                for s in range(seeds):
                    rng = np.random.default_rng(10_000 * B + s)
                    kw = {"peer_optima": peers} if name == "transfer" else {}
                    pick, _q = fn(rp, hp, B, rng, **kw)
                    reg.append(100.0 * (best - rp.true_asr(pick)))
                    blocks.setdefault((name, B), {}).setdefault(s, []).append(reg[-1])
                rows.append({"grid": label, "model": m, "method": name,
                             "budget": B, "regret": float(np.mean(reg))})
    paired = {k: np.array([np.mean(v[s]) for s in sorted(v)])
              for k, v in blocks.items()}
    return pd.DataFrame(rows), paired


def holm(pvals):
    pvals = np.asarray(pvals, float)
    order, adj, run_max, n = np.argsort(pvals), np.empty_like(pvals), 0.0, len(pvals)
    for rank, i in enumerate(order):
        run_max = max(run_max, (n - rank) * pvals[i])
        adj[i] = min(1.0, run_max)
    return adj


def gate(pdf, published, budgets, seeds, hp):
    """Every method, every target, against the published per-seed means."""
    pmodels = [m for m in POOL if m in set(pdf.model.unique())]
    if len(pmodels) != len(POOL):
        sys.exit(f"probe cache is missing {set(POOL) - set(pmodels)}")
    want = (published[published.budget.isin(budgets)]
            .pivot_table(index=["model", "method"], columns="budget",
                         values="regret", aggfunc="mean") * 100.0)
    got, _ = run(pdf, pmodels, budgets, seeds, hp, "probe",
                 methods={k: METHODS[k] for k in ANCHORED}, quiet=True)
    bad = []
    print(f"  {'method':<20}{'target':<26}" +
          "".join(f"{('B=%d' % b):>18}" for b in budgets))
    for name in ANCHORED:
        for m in pmodels:
            cells = []
            for B in budgets:
                g = float(got[(got.method == name) & (got.model == m)
                              & (got.budget == B)].regret.iloc[0])
                try:
                    w = float(want.loc[(m, name), B])
                except KeyError:
                    cells.append("  (not in file)")
                    continue
                ok = abs(g - w) <= TOL
                cells.append(f"{g:>8.2f}/{w:<6.2f}{'' if ok else ' X'}")
                if not ok:
                    bad.append((name, m, B, g, w))
            print(f"  {name:<20}{m:<26}" + "".join(f"{c:>18}" for c in cells))
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/eval_half")
    ap.add_argument("--probe", default=None)
    ap.add_argument("--published", default="results/per_seed_regrets.csv")
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[120, 240, 360, 480, 720, 960])
    ap.add_argument("--gate-budgets", type=int, nargs="+", default=[240, 960])
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--matched", action="store_true",
                    help="also replay the PROBE grid over the same four "
                         "targets, so probe and held-out numbers are "
                         "comparable; Table 6 averages five, including the "
                         "API-served target, which has no held-out grid")
    ap.add_argument("--out", default="results/heldout_replay")
    a = ap.parse_args()
    hp = Hyper()

    probe = a.probe or str(CACHE)
    for p in (probe, a.published):
        if not os.path.exists(p):
            sys.exit(f"not found: {p}")
    pdf = pd.read_parquet(probe)
    if "success" not in pdf.columns:
        c = next((x for x in ("success_strict", "y") if x in pdf.columns), None)
        if c is None:
            sys.exit(f"probe cache has no success column; has {list(pdf.columns)}")
        pdf = pdf.rename(columns={c: "success"})
    published = pd.read_csv(a.published)

    print(f"\ngate: every method against {a.published}, live / published\n")
    bad = gate(pdf, published, a.gate_budgets, a.seeds, hp)
    if bad:
        print("\nThe gate failed:")
        for name, m, B, g, w in bad:
            print(f"  {name} on {m} at B={B}: {g:.2f} live, {w:.2f} published")
        sys.exit("\nDo not trust any held-out number from this run. Check that\n"
                 "--seeds matches the published file and that each method maps\n"
                 "to the implementation that produced it. In particular\n"
                 "libra_select.bl_race_only is 'race_only_legacy', not the\n"
                 "full-space halving of Section V.")
    print("\n  gate passed: every method reproduces the published per-seed means\n")
    if a.check:
        return 0

    print(f"loading the held-out grid from {a.root}")
    hdf = load_heldout(a.root)
    print(f"  {len(hdf):,} rows over {hdf.model.nunique()} targets\n")
    res, paired = run(hdf, OPEN_WEIGHT, a.budgets, a.seeds, hp, "heldout")

    from scipy import stats
    comp = []
    for B in a.budgets:
        d = paired[(RACE, B)] - paired[("LIBRA", B)]      # >0 favors LIBRA-S
        t, p = stats.ttest_rel(paired[(RACE, B)], paired[("LIBRA", B)])
        comp.append({"budget": B, "diff": float(d.mean()),
                     "se": float(d.std(ddof=1) / np.sqrt(len(d))),
                     "t": float(t), "p_raw": float(p)})
    for row, padj in zip(comp, holm([r["p_raw"] for r in comp])):
        row["p_holm"] = float(padj)

    piv = res.pivot_table(index="budget", columns="method",
                          values="regret", aggfunc="mean")
    print(f"\n{'='*78}\nHELD-OUT mean regret (points), 4 open-weight targets, "
          f"{a.seeds} seeds\n")
    print(piv.round(2).to_string())
    print(f"\n{'='*78}\nLIBRA-S vs full-space halving on held-out instructions\n")
    print(f"  {'budget':>7}{'diff':>9}{'se':>7}{'p_holm':>10}")
    for r in comp:
        print(f"  {r['budget']:>7}{r['diff']:>9.2f}{r['se']:>7.2f}"
              f"{r['p_holm']:>10.4f}{' *' if r['p_holm'] < 0.05 else ''}")
    print(f"\n  diff > 0 favors LIBRA-S. * survives Holm across {len(comp)} budgets.")

    print(f"\n{'='*78}\nPer target at the two headline budgets\n")
    for B in [b for b in (240, 960) if b in a.budgets]:
        print(f"  budget {B}")
        print(res[res.budget == B].pivot_table(index="model", columns="method",
                                               values="regret").round(2).to_string(), "\n")

    matched = None
    if a.matched:
        print(f"{'='*78}\nMATCHED: the same four targets on both grids\n")
        mres, mpaired = run(pdf, OPEN_WEIGHT, a.budgets, a.seeds, hp,
                            "probe4", quiet=True)
        mp = mres.pivot_table(index="budget", columns="method", values="regret")
        print(f"  {'':>7}{'LIBRA-S':>20}{'':>6}{'full-space halving':>26}")
        print(f"  {'budget':>7}{'probe':>10}{'heldout':>10}{'delta':>6}"
              f"{'probe':>13}{'heldout':>10}{'delta':>7}")
        for B in a.budgets:
            pl, hl = mp.loc[B, "LIBRA"], piv.loc[B, "LIBRA"]
            pr, hr = mp.loc[B, RACE], piv.loc[B, RACE]
            print(f"  {B:>7}{pl:>10.2f}{hl:>10.2f}{hl-pl:>+6.2f}"
                  f"{pr:>13.2f}{hr:>10.2f}{hr-pr:>+7.2f}")
        mcomp = []
        for B in a.budgets:
            t, p = stats.ttest_rel(mpaired[(RACE, B)], mpaired[("LIBRA", B)])
            mcomp.append({"budget": B, "t": float(t), "p_raw": float(p),
                          "diff": float((mpaired[(RACE, B)]
                                         - mpaired[("LIBRA", B)]).mean())})
        for row, padj in zip(mcomp, holm([r["p_raw"] for r in mcomp])):
            row["p_holm"] = float(padj)
        print("\n  LIBRA-S vs full-space halving, PROBE grid, same four targets:")
        print(f"  {'budget':>7}{'diff':>9}{'p_holm':>10}")
        for r in mcomp:
            print(f"  {r['budget']:>7}{r['diff']:>9.2f}{r['p_holm']:>10.4f}"
                  f"{' *' if r['p_holm'] < 0.05 else ''}")
        print("\n  A budget significant on one grid and not the other is the\n"
              "  finding to report, not the raw regret difference.")
        matched = {"probe_four_targets": mp.round(4).to_dict(),
                   "libra_vs_halving_probe_four": mcomp}

    os.makedirs(a.out, exist_ok=True)
    out = {"seeds": a.seeds, "budgets": a.budgets, "targets": OPEN_WEIGHT,
           "n_instructions": N_INSTR, "n_configurations": N_CONFIG,
           "gate": {"source": a.published, "budgets": a.gate_budgets,
                    "tolerance_points": TOL, "methods": ANCHORED},
           "heldout_mean_regret": piv.round(4).to_dict(),
           "libra_vs_halving": comp,
           "matched": matched,
           "per_target": res.to_dict(orient="records"),
           "scope": ("four open-weight targets; the API-served target has no "
                     "held-out grid. The screen and component pools were fixed "
                     "on probe data.")}
    p = os.path.join(a.out, "heldout_replay.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")
    print("No generations are read or written: this replays stored labels only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
