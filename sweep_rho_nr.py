#!/usr/bin/env python3
"""
sweep_rho_nr.py -- sensitivity of LIBRA-S to the budget split (screen_frac)
and the racing start (n_race0). Answers Martin's "How were rho and n_r chosen?"

Written against the repo's own API and modelled on table6_paired.py, so the
protocol matches Section V-D exactly: one block per seed, each block the mean
over the five targets; seeds 10000B+s; two-sided paired t-tests; Holm across
every comparison in the table.

libra_select.sensitivity() already sweeps screen_frac, but at a single budget,
with seeds rng(s) rather than the published 10000B+s rule, and with no paired
test. It also does not touch n_race0 at all. This covers both parameters over
the full budget range under the published protocol.

Run from the repo root:

    python sweep_rho_nr.py --check          # reproduce the anchors only
    python sweep_rho_nr.py --out results/rho_nr_sweep
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

BUDGETS = [120, 240, 360, 480, 720, 960, 1440]   # matches table6_paired.py
SEEDS = 200
FRAC_FIXED, NR_FIXED = 0.7, 4
FRAC_GRID = [0.5, 0.6, 0.7, 0.8]
NR_GRID = [2, 4, 8]

# Published Table 6, column S-dagger. Nothing is swept until these reproduce.
ANCHORS = {240: 6.43, 960: 2.29}
ANCHOR_TOL = 0.02


def label(frac, nr):
    return f"frac={frac},n_race0={nr}"


def run_setting(df, Replay, Hyper, select, SCREEN_DESIGN, POOL,
                frac, nr, budgets, seeds):
    """blocks[budget] = per-seed mean regret over the five targets."""
    cells = list(SCREEN_DESIGN)
    hp = Hyper(k_screen=len(cells), screen_frac=frac, n_race0=nr)
    out, spend = {}, {}
    for B in budgets:
        per_seed = np.zeros((seeds, len(POOL)))
        q_used = []
        for j, m in enumerate(POOL):
            rp = Replay(df, m)
            _, star = rp.optimum()
            for s in range(seeds):
                pick, q = select(rp, hp, B,
                                 np.random.default_rng(10_000 * B + s),
                                 screen_cells=cells)
                per_seed[s, j] = (star - rp.true_asr(pick)) * 100
                q_used.append(int(q))
        out[B] = per_seed.mean(axis=1)
        spend[B] = (int(np.max(q_used)), float(np.mean(q_used)))
    return out, spend


def paired_t(a, b):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    sd = d.std(ddof=1)
    if n < 2 or sd == 0:
        return 1.0
    t = d.mean() / (sd / math.sqrt(n))
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), n - 1))
    except Exception:                                    # noqa: BLE001
        return float(math.erfc(abs(t) / math.sqrt(2)))


def holm(p):
    items = sorted(p.items(), key=lambda kv: kv[1])
    m, adj, run = len(items), {}, 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        adj[k] = run
    return adj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--out", default="results/rho_nr_sweep")
    a = ap.parse_args()

    root = pathlib.Path.cwd()
    if not (root / "libra_select.py").exists():
        print("run this from the root of the LIBRA tree", file=sys.stderr)
        return 2
    sys.path.insert(0, ".")
    from libra_select import (Replay, Hyper, select, SCREEN_DESIGN,  # noqa: E402
                             POOL, CACHE)
    df = pd.read_parquet(CACHE)

    # -- anchors ------------------------------------------------------------
    print(f"reproducing Table 6 S-dagger at screen_frac={FRAC_FIXED}, "
          f"n_race0={NR_FIXED}, {a.seeds} seeds:")
    base, base_spend = run_setting(df, Replay, Hyper, select, SCREEN_DESIGN,
                                   POOL, FRAC_FIXED, NR_FIXED,
                                   sorted(ANCHORS), a.seeds)
    bad = []
    for B, want in sorted(ANCHORS.items()):
        got = base[B].mean()
        ok = abs(got - want) <= ANCHOR_TOL
        print(f"  budget {B:4d}   published {want:5.2f}   recomputed {got:6.3f}"
              f"   {'ok' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(B)
    if bad:
        print(f"\nStopping: recomputed misses published at {bad}.",
              file=sys.stderr)
        return 1
    print("  anchors reproduce.")
    if a.check:
        print("\n--check: nothing swept, nothing written.")
        return 0

    # -- sweep, one parameter at a time -------------------------------------
    settings = ([(f, NR_FIXED) for f in FRAC_GRID]
                + [(FRAC_FIXED, n) for n in NR_GRID if n != NR_FIXED])
    baseline = label(FRAC_FIXED, NR_FIXED)
    print(f"\nsweeping {len(settings)} settings x {len(BUDGETS)} budgets x "
          f"{a.seeds} seeds x {len(POOL)} targets = "
          f"{len(settings)*len(BUDGETS)*a.seeds*len(POOL):,} replays")

    blocks, spends = {}, {}
    for frac, nr in settings:
        b, sp = run_setting(df, Replay, Hyper, select, SCREEN_DESIGN, POOL,
                            frac, nr, BUDGETS, a.seeds)
        lab = label(frac, nr)
        for B in BUDGETS:
            blocks[(lab, B)] = b[B]
            spends[(lab, B)] = sp[B]
        print(f"  {lab:<22} done")

    raw = {(lab, B): paired_t(blocks[(lab, B)], blocks[(baseline, B)])
           for lab in {label(f, n) for f, n in settings} if lab != baseline
           for B in BUDGETS}
    adj = holm(raw)

    # -- report -------------------------------------------------------------
    labs = [label(f, n) for f, n in settings]
    print(f"\n{'setting':<24}" + "".join(f"{B:>9}" for B in BUDGETS))
    for lab in labs:
        mark = "   <- production" if lab == baseline else ""
        row = "".join(
            f"{blocks[(lab,B)].mean():8.2f}"
            + ("*" if adj.get((lab, B), 1) < 0.05 else " ")
            for B in BUDGETS)
        print(f"{lab:<24}{row}{mark}")
    print("  * = differs from production after Holm across "
          f"{len(raw)} comparisons")

    print("\nmax unique queries spent against the nominal budget:")
    print(f"{'setting':<24}" + "".join(f"{B:>9}" for B in BUDGETS))
    for lab in labs:
        print(f"{lab:<24}" + "".join(f"{spends[(lab,B)][0]:9d}"
                                     for B in BUDGETS))

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
    except Exception:                                    # noqa: BLE001
        sha = "unknown"
    rows = [{"screen_frac": f, "n_race0": n, "budget": B,
             "mean_regret": float(blocks[(label(f, n), B)].mean()),
             "se": float(blocks[(label(f, n), B)].std(ddof=1)
                         / math.sqrt(a.seeds)),
             "max_queries": spends[(label(f, n), B)][0],
             "mean_queries": spends[(label(f, n), B)][1],
             "p_raw": raw.get((label(f, n), B)),
             "p_holm": adj.get((label(f, n), B)),
             "significant": bool(adj.get((label(f, n), B), 1) < 0.05)}
            for f, n in settings for B in BUDGETS]
    (out / "rho_nr_sweep.json").write_text(json.dumps(
        {"git_sha": sha, "seeds": a.seeds, "pool": list(POOL),
         "budgets": BUDGETS, "baseline": baseline,
         "screen_design_sha256": hashlib.sha256(
             "\n".join(sorted(SCREEN_DESIGN)).encode()).hexdigest(),
         "anchors": ANCHORS, "rows": rows}, indent=2))
    print(f"\n  wrote {out/'rho_nr_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
