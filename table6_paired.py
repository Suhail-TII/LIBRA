#!/usr/bin/env python3
"""
table6_paired.py -- the screening-size sweep with per-seed export, frozen
design lists, and paired comparisons computed from those same runs.

Why this exists rather than reading someone else's paired tests: a design is
an experimental input. Two independently constructed 22-cell designs agree to
within 0.03 points, which is a useful sensitivity replication, but they are
different cell lists. Every interval and test printed here is computed from
the per-seed outputs of the exact lists frozen in table6_designs.json.

Comparisons are paired over seed blocks. Each block averages the five targets
under one seed, so the pairing removes the between-target variance that would
otherwise swamp a half-point difference. Holm adjustment is applied across all
design x budget comparisons reported.

Read-only with respect to results/ except for the three files it writes.

    python table6_paired.py --seeds 200
    python table6_paired.py --seeds 20 --quick     # shape check first
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

BUDGETS = [120, 240, 360, 480, 720, 960, 1440]


def coverage_design(per_framing, F, T, O, seed=0, trials=200_000):
    """
    Maximise distinct (framing, template) pairs, then (framing, order) pairs,
    then evenness of template and order counts. Deterministic given seed.
    """
    rng = np.random.default_rng(seed)
    opts = [(t, o) for t in T for o in O]
    best, best_s = None, (-1, -1, -1e9)
    for _ in range(trials):
        d = [(f, *opts[i]) for f in F
             for i in rng.choice(len(opts), size=per_framing, replace=False)]
        ft = len({(c[0], c[1]) for c in d})
        fo = len({(c[0], c[2]) for c in d})
        tc, oc = Counter(c[1] for c in d), Counter(c[2] for c in d)
        bal = -(np.std([tc[x] for x in T]) + np.std([oc[x] for x in O]))
        if (ft, fo, bal) > best_s:
            best, best_s = d, (ft, fo, bal)
    return [f"{a}|{b}|{c}" for a, b, c in best]


def coverage_of(cells):
    ft = len({tuple(c.split("|")[:2]) for c in cells})
    fo = len({(c.split("|")[0], c.split("|")[2]) for c in cells})
    return ft, fo


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    n = len(pvals)
    idx = np.argsort(pvals)
    out = np.empty(n)
    running = 0.0
    for rank, i in enumerate(idx):
        adj = (n - rank) * pvals[i]
        running = max(running, adj)
        out[i] = min(1.0, running)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--quick", action="store_true",
                    help="fewer construction trials, for a shape check")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    sys.path.insert(0, ".")

    from libra_select import Replay, Hyper, select, SCREEN_DESIGN, POOL, CACHE
    # both constants come straight from attacks.libra_framings. Going via
    # run_stage_b pulled in analysis.metrics and therefore torch, so Table VI
    # could not be reproduced without the GPU stack despite being pure replay
    # over CSVs. run_stage_b only re-exports the name anyway.
    from attacks.libra_framings import UTOS_TOP4, STAGE_B_FRAMINGS
    from scipy import stats

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(CACHE)

    F, T = sorted(STAGE_B_FRAMINGS), sorted(UTOS_TOP4)
    O = ["libra_then_utos", "utos_then_libra"]
    trials = 20_000 if a.quick else 200_000

    designs = {f"{11*n}_coverage": coverage_design(n, F, T, O, trials=trials)
               for n in (1, 2, 3, 4)}
    designs["22_main"] = list(SCREEN_DESIGN)

    # freeze the exact lists, with a hash per design, before anything uses them
    frozen = {}
    for k, cells in designs.items():
        ft, fo = coverage_of(cells)
        frozen[k] = {
            "n_cells": len(cells),
            "framing_template_pairs": ft,
            "framing_order_pairs": fo,
            "sha256": hashlib.sha256("\n".join(sorted(cells)).encode()).hexdigest(),
            "cells": sorted(cells),
        }
    (out / "table6_designs.json").write_text(json.dumps(frozen, indent=2))

    print("    design        cells   framing-template   framing-order   sha256")
    for k in sorted(frozen, key=lambda x: (frozen[x]["n_cells"], x)):
        d = frozen[k]
        tag = "  <- production" if k == "22_main" else ""
        print(f"    {k:<13} {d['n_cells']:>5}   {d['framing_template_pairs']:>10}/44"
              f"   {d['framing_order_pairs']:>10}/22   {d['sha256'][:12]}{tag}")

    # ---- the sweep, exporting every run -----------------------------------
    rows = []
    for B in BUDGETS:
        for k, cells in designs.items():
            hp = Hyper(k_screen=len(cells))
            for m in POOL:
                rp = Replay(df, m)
                _, star = rp.optimum()
                for s in range(a.seeds):
                    pick, q = select(rp, hp, B, np.random.default_rng(10_000 * B + s),
                                     screen_cells=cells)
                    rows.append(dict(budget=B, design=k, model=m, seed=s,
                                     rng_seed=10_000 * B + s, selected=pick,
                                     queries=int(q),
                                     regret_points=float(star - rp.true_asr(pick)) * 100))
        print(f"    budget {B:>4} done")
    ps = pd.DataFrame(rows)
    ps.to_csv(out / "table6_per_seed.csv", index=False)

    # ---- the table ---------------------------------------------------------
    order = sorted(designs, key=lambda x: (len(designs[x]), x))
    means = ps.pivot_table(index="budget", columns="design",
                           values="regret_points")[order]
    print("\nTABLE 6  mean regret in points\n")
    print(means.round(2).to_string())

    # ---- paired comparisons, from these runs only -------------------------
    # one block per seed, each block the mean over the five targets
    blocks = ps.groupby(["budget", "design", "seed"]).regret_points.mean().unstack("design")
    recs = []
    for B in BUDGETS:
        b = blocks.loc[B]
        for k in order:
            if k == "22_main":
                continue
            diff = (b[k] - b["22_main"]).to_numpy()
            t, p = stats.ttest_1samp(diff, 0.0)
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            crit = stats.t.ppf(0.975, len(diff) - 1)
            recs.append(dict(budget=B, comparison=f"{k} minus 22_main",
                             mean_difference_points=float(diff.mean()),
                             se_points=float(se),
                             ci95_lo=float(diff.mean() - crit * se),
                             ci95_hi=float(diff.mean() + crit * se),
                             t=float(t), df=len(diff) - 1,
                             raw_p=float(p), blocks=len(diff)))
    comp = pd.DataFrame(recs)
    comp["holm_p"] = holm(comp.raw_p.to_numpy())
    comp.to_csv(out / "table6_paired.csv", index=False)

    print("\nPAIRED COMPARISONS vs the production design "
          f"({len(comp)} comparisons, Holm-adjusted)\n")
    print(f"{'budget':>7} {'comparison':<24}{'diff':>8}{'95% CI':>18}"
          f"{'raw p':>10}{'Holm p':>9}  verdict")
    for _, r in comp.iterrows():
        v = ("better" if r.holm_p < 0.05 and r.mean_difference_points < 0 else
             "worse" if r.holm_p < 0.05 and r.mean_difference_points > 0 else
             "no detectable difference")
        print(f"{int(r.budget):>7} {r.comparison:<24}{r.mean_difference_points:>8.3f}"
              f"  [{r.ci95_lo:>6.2f},{r.ci95_hi:>6.2f}]{r.raw_p:>10.4f}"
              f"{r.holm_p:>9.4f}  {v}")

    print("\n  Holm adjustment is across all comparisons in this table.")
    print("  'no detectable difference' is not evidence of equivalence.")
    print(f"\n  wrote {out}/table6_designs.json, table6_per_seed.csv, table6_paired.csv")


if __name__ == "__main__":
    main()