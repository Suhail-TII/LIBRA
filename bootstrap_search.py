#!/usr/bin/env python3
"""
bootstrap_search.py -- V 12 / V 19: instruction-level uncertainty for the
Section V search comparison.

Section V currently pairs over 200 search seeds. Seed p-values shrink as seeds
are added and describe replay randomness, not generalization. The sampling that
actually happened is the draw of AdvBench instructions, so the uncertainty that
matters is over those.

reviewer_analyses.bootstrap() already resamples instructions, but only for the
descriptive quantities (transfer cost, order effect, c* ASR), and it holds c*
fixed across resamples. This does the part that is missing: resample the
instruction grid, RE-PICK c* inside each resample, replay both methods against
that resample's own optimum, and report the paired difference over resamples.

    python bootstrap_search.py --check
    python bootstrap_search.py --boot 200 --seeds 10 --out results/boot_search
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

BUDGETS = [120, 240, 360, 480, 720, 960]
# Table 6 S-dagger / make_tables.py, five-target means. Reproduced first.
ANCHORS = {240: 6.43, 960: 2.29}
FSH_PUBLISHED = {120: 40.36, 240: 7.95, 360: 5.11, 480: 3.70, 720: 3.00, 960: 2.57}
TOL = 0.02


def resampled(base, cols):
    """A Replay over the same configs but a bootstrap draw of its columns.

    true_asr() reads _m[i][_seen[i]], and optimum() is computed from it, so
    each resample gets its own c* with no further work.
    """
    r = copy.copy(base)
    r._m = base._m[:, cols]
    r._seen = base._seen[:, cols]
    r.instructions = np.arange(len(cols))
    r._jdx = {n: j for j, n in enumerate(r.instructions)}
    r.asked, r.n_queries = set(), 0
    return r


def one_resample(bases, draw, budgets, seeds, select, halving, Hyper, cells):
    """Mean regret over the five targets, per method and budget, for one draw."""
    hp = Hyper(k_screen=len(cells))
    reps = {}
    for m, base in bases.items():
        have = {int(x): j for j, x in enumerate(base.instructions)}
        cols = np.array([have[i] for i in draw if i in have])
        rp = resampled(base, cols)
        _, star = rp.optimum()
        for B in budgets:
            for name, fn in (("libra_s", select), ("fsh", halving)):
                acc = []
                for s in range(seeds):
                    rng = np.random.default_rng(10_000 * B + s)
                    kw = {"screen_cells": cells} if name == "libra_s" else {}
                    pick, _ = fn(rp, hp, B, rng, **kw)
                    acc.append((star - rp.true_asr(pick)) * 100)
                reps.setdefault((name, B), []).append(np.mean(acc))
    return {k: float(np.mean(v)) for k, v in reps.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results/boot_search")
    a = ap.parse_args()

    if not pathlib.Path("libra_select.py").exists():
        print("run this from the root of the LIBRA tree", file=sys.stderr)
        return 2
    sys.path.insert(0, ".")
    from libra_select import (Replay, Hyper, select, SCREEN_DESIGN,  # noqa: E402
                              POOL, CACHE)
    from full_space_halving import bl_successive_halving  # noqa: E402

    df = pd.read_parquet(CACHE)
    cells = list(SCREEN_DESIGN)
    bases = {m: Replay(df, m) for m in POOL}

    # -- anchors on the real grid, 200 seeds, before any resampling ---------
    print("reproducing the published five-target means on the full grid:")
    hp = Hyper(k_screen=len(cells))
    bad = []
    for B in sorted(ANCHORS):
        for name, fn, want in (("LIBRA-S", select, ANCHORS[B]),
                               ("full-space halving", bl_successive_halving,
                                FSH_PUBLISHED[B])):
            vals = []
            for m, rp in bases.items():
                _, star = rp.optimum()
                kw = {"screen_cells": cells} if name == "LIBRA-S" else {}
                vals += [(star - rp.true_asr(
                    fn(rp, hp, B, np.random.default_rng(10_000 * B + s), **kw)[0])) * 100
                    for s in range(200)]
            got = float(np.mean(vals))
            ok = abs(got - want) <= TOL
            print(f"  {name:<20} budget {B:4d}  published {want:6.2f}"
                  f"  recomputed {got:7.3f}  {'ok' if ok else 'MISMATCH'}")
            if not ok:
                bad.append((name, B))
    if bad:
        print(f"\nStopping: {bad} did not reproduce.", file=sys.stderr)
        return 1
    print("  anchors reproduce.")
    if a.check:
        print("\n--check: nothing resampled, nothing written.")
        return 0

    # -- the bootstrap ------------------------------------------------------
    ids = sorted({int(x) for b in bases.values() for x in b.instructions})
    rng = np.random.default_rng(20261109)
    print(f"\n{a.boot} instruction resamples x {len(BUDGETS)} budgets x "
          f"{a.seeds} seeds x {len(POOL)} targets x 2 methods = "
          f"{a.boot*len(BUDGETS)*a.seeds*len(POOL)*2:,} replays")
    print("c* is re-picked inside every resample.\n")

    rows = {}
    for b in range(a.boot):
        draw = rng.choice(ids, size=len(ids), replace=True).tolist()
        out = one_resample(bases, draw, BUDGETS, a.seeds, select,
                           bl_successive_halving, Hyper, cells)
        for k, v in out.items():
            rows.setdefault(k, []).append(v)
        if (b + 1) % 25 == 0:
            print(f"    {b+1}/{a.boot} resamples", flush=True)

    arr = {k: np.asarray(v) for k, v in rows.items()}
    q = lambda x: (np.percentile(x, 2.5), np.percentile(x, 97.5))  # noqa: E731

    print(f"\n{'budget':>7}{'LIBRA-S':>24}{'full-space halving':>26}"
          f"{'FSH - LIBRA-S':>22}{'P(L<F)':>9}")
    res = []
    for B in BUDGETS:
        L, F = arr[("libra_s", B)], arr[("fsh", B)]
        d = F - L
        lo, hi = q(d)
        win = float((d > 0).mean())
        excl = "yes" if (lo > 0 or hi < 0) else "no"
        print(f"{B:>7}  {L.mean():7.2f} [{q(L)[0]:5.2f},{q(L)[1]:6.2f}]"
              f"  {F.mean():8.2f} [{q(F)[0]:5.2f},{q(F)[1]:6.2f}]"
              f"  {d.mean():+7.2f} [{lo:+6.2f},{hi:+6.2f}]{win:>9.1%}")
        res.append(dict(budget=B, libra_s_mean=float(L.mean()),
                        libra_s_ci=[float(x) for x in q(L)],
                        fsh_mean=float(F.mean()),
                        fsh_ci=[float(x) for x in q(F)],
                        diff_mean=float(d.mean()),
                        diff_ci=[float(lo), float(hi)],
                        p_libra_better=win,
                        interval_excludes_zero=(excl == "yes")))
    print("\n  intervals are 2.5/97.5 percentiles over instruction resamples")
    print("  P(L<F) = share of resamples in which LIBRA-S had lower regret")
    print("\n  interval excludes zero at: "
          + (", ".join(str(r["budget"]) for r in res
                       if r["interval_excludes_zero"]) or "no budget"))

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
    except Exception:                                      # noqa: BLE001
        sha = "unknown"
    (out / "boot_search.json").write_text(json.dumps(
        {"git_sha": sha, "n_boot": a.boot, "seeds_per_resample": a.seeds,
         "targets": list(POOL), "budgets": BUDGETS,
         "cstar_repicked_per_resample": True,
         "anchors": {"libra_s": ANCHORS, "full_space_halving": FSH_PUBLISHED},
         "rows": res}, indent=2))
    print(f"\n  wrote {out/'boot_search.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
