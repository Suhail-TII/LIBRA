#!/usr/bin/env python3
"""
transfer_oob.py -- selection bias in the transfer gap of Section IV-A.

Each target's optimum is the maximum of 103 noisy estimates, so scoring it on
the instructions that selected it is optimistic. This quantifies that, for the
target's own optimum and for the transfer gap, by bootstrapping the probe
instructions and re-picking every optimum inside each resample.

    python transfer_oob.py --cache results/replay_cache.parquet --check
    python transfer_oob.py --cache results/replay_cache.parquet \
        --boot 2000 --seed 20261115 --out results/transfer_oob/transfer_oob.json

Two donor policies are reported, because the comparison is only meaningful if
the SAME policy is used in both arms:

  reselected  every target's optimum, donors included, is re-picked on that
              target's own in-bag draw. This is the protocol the paper reports.
  fixed       donors are held at their full-grid optima in both arms, so only
              the recipient's optimum is re-picked.

Mixing the two, for instance full-grid donors in the in-bag arm and in-bag
donors in the out-of-bag arm, measures a change of donor policy on top of the
change of scoring set and does not isolate the selection bias. An earlier
analysis of ours made exactly that mistake and reported a spurious
cancellation; the two policies are both reported here so the difference is
visible rather than assumed.

Scoring: the recipient's own optimum and its donors' configurations are both
evaluated on the recipient's instructions, using the same in-bag or
out-of-bag partition. Ties in the argmax are broken by sorted configuration
order. Nothing here reads the manuscript; --check compares against the values
reported in Section IV-A.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

TARGETS = ["Llama-2-7B-Chat", "Llama-3.1-8B-Instruct", "Qwen3-14B",
           "gpt-5.1", "Ministral-3-8B-Instruct"]

# Section IV-A, to two decimals. --check verifies these.
PAPER = {"gap_full_grid": 32.6, "gap_oob": 33.3, "gap_optimism": 1.9,
         "oob_ci": (27.1, 39.8)}
TOL = 0.3


def load(path):
    df = pd.read_parquet(path)
    df = df[df.model.isin(TARGETS)]
    if df.duplicated(["model", "config", "instruction_id"]).any():
        sys.exit("repeated (model, config, instruction) keys in the cache")
    ids = np.array(sorted(df.instruction_id.unique()))
    configs = sorted(df.config.unique())
    if len(configs) != 103:
        sys.exit(f"expected 103 configurations, found {len(configs)}")
    grids, masks = [], []
    for m in TARGETS:
        g = (df[df.model == m]
             .pivot(index="config", columns="instruction_id", values="success")
             .reindex(index=configs, columns=ids))
        present = g.notna().sum(axis=0).to_numpy()
        if not np.isin(present, [0, len(configs)]).all():
            sys.exit(f"partially measured instruction column for {m}")
        masks.append(present == len(configs))
        grids.append(g.fillna(False).to_numpy(dtype=float))
    return np.stack(grids), np.stack(masks), ids, configs


def run(grids, masks, policy, n_boot, seed):
    """Return per-replicate (gap_in, gap_oob, own_in, own_oob, peer_in, peer_oob)."""
    rng = np.random.default_rng(seed)
    n_t = len(TARGETS)
    full = np.stack([g[:, k].mean(axis=1) for g, k in zip(grids, masks)])
    star_full = full.argmax(axis=1)

    inb, oob = [], []
    for g, k in zip(grids, masks):
        pos = np.flatnonzero(k)
        w = np.array([np.bincount(rng.choice(pos, len(pos), replace=True),
                                  minlength=len(k)) for _ in range(n_boot)])
        o = ((w == 0) & k).astype(float)
        if np.any(o.sum(axis=1) == 0):
            sys.exit("a replicate left no out-of-bag instructions")
        inb.append(w @ g.T / w.sum(axis=1)[:, None])
        oob.append(o @ g.T / o.sum(axis=1)[:, None])
    inb, oob = np.stack(inb), np.stack(oob)

    chosen = inb.argmax(axis=2)          # in-bag pick, per target per replicate
    r = np.arange(n_boot)
    own_in = np.stack([inb[m, r, chosen[m]] for m in range(n_t)])
    own_oob = np.stack([oob[m, r, chosen[m]] for m in range(n_t)])

    def donor(m, d):
        return chosen[d] if policy == "reselected" else star_full[d]

    peer_in = np.stack([np.mean([inb[m, r, donor(m, d)]
                                 for d in range(n_t) if d != m], axis=0)
                        for m in range(n_t)])
    peer_oob = np.stack([np.mean([oob[m, r, donor(m, d)]
                                  for d in range(n_t) if d != m], axis=0)
                         for m in range(n_t)])
    return own_in, own_oob, peer_in, peer_oob, full, star_full


def summarize(own_in, own_oob, peer_in, peer_oob):
    gi = (own_in - peer_in).mean(axis=0) * 100
    go = (own_oob - peer_oob).mean(axis=0) * 100
    return {
        "own_asr_in": float(own_in.mean() * 100),
        "own_asr_oob": float(own_oob.mean() * 100),
        "own_asr_optimism": float((own_in - own_oob).mean() * 100),
        "peer_asr_in": float(peer_in.mean() * 100),
        "peer_asr_oob": float(peer_oob.mean() * 100),
        "peer_asr_optimism": float((peer_in - peer_oob).mean() * 100),
        "gap_in": float(gi.mean()),
        "gap_oob": float(go.mean()),
        "gap_optimism": float((gi - go).mean()),
        "gap_oob_ci95": [float(x) for x in np.percentile(go, [2.5, 97.5])],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=pathlib.Path,
                    default=pathlib.Path("results/replay_cache.parquet"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("results/transfer_oob/transfer_oob.json"))
    ap.add_argument("--check", action="store_true",
                    help="compare the reselected-donor result against Section IV-A")
    a = ap.parse_args()
    if not a.cache.exists():
        sys.exit(f"missing {a.cache}")

    grids, masks, ids, configs = load(a.cache)
    digest = hashlib.sha256(a.cache.read_bytes()).hexdigest()

    out = {
        "cache": str(a.cache), "cache_sha256": digest,
        "seed": a.seed, "n_boot": a.boot, "targets": TARGETS,
        "instructions_per_target": dict(zip(TARGETS, masks.sum(axis=1).tolist())),
        "n_configurations": len(configs),
        "units": "ASR in percent; gaps and optimism in percentage points",
    }
    for policy in ("reselected", "fixed"):
        oi, oo, pi, po, full, star = run(grids, masks, policy, a.boot, a.seed)
        out[policy] = summarize(oi, oo, pi, po)

    own = np.array([full[m, star[m]] for m in range(len(TARGETS))])
    peer = np.array([np.mean([full[m, star[d]] for d in range(len(TARGETS)) if d != m])
                     for m in range(len(TARGETS))])
    out["full_grid"] = {"own_asr": float(own.mean() * 100),
                        "peer_asr": float(peer.mean() * 100),
                        "gap": float((own - peer).mean() * 100),
                        "optima": {t: configs[star[i]] for i, t in enumerate(TARGETS)}}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")

    w, f, g = out["reselected"], out["fixed"], out["full_grid"]
    print(f"\ncache {a.cache}  sha256 {digest[:16]}...  seed {a.seed}  "
          f"{a.boot} resamples\n")
    print(f"  full grid, optima fixed        gap {g['gap']:.2f}\n")
    print(f"  {'':<16}{'own in':>9}{'own oob':>9}{'gap in':>9}{'gap oob':>9}{'optimism':>10}")
    for name, d in (("reselected donors", w), ("fixed donors", f)):
        print(f"  {name:<16}{d['own_asr_in']:>9.2f}{d['own_asr_oob']:>9.2f}"
              f"{d['gap_in']:>9.2f}{d['gap_oob']:>9.2f}{d['gap_optimism']:>10.2f}")
    print(f"\n  out-of-bag gap 95% interval, reselected donors: "
          f"[{w['gap_oob_ci95'][0]:.1f}, {w['gap_oob_ci95'][1]:.1f}]")
    print(f"  peer-ASR optimism: reselected {w['peer_asr_optimism']:+.2f}, "
          f"fixed {f['peer_asr_optimism']:+.2f}")
    print(f"\n  wrote {a.out}")

    if a.check:
        rows = [("full-grid gap", g["gap"], PAPER["gap_full_grid"]),
                ("out-of-bag gap", w["gap_oob"], PAPER["gap_oob"]),
                ("gap optimism", w["gap_optimism"], PAPER["gap_optimism"]),
                ("interval low", w["gap_oob_ci95"][0], PAPER["oob_ci"][0]),
                ("interval high", w["gap_oob_ci95"][1], PAPER["oob_ci"][1])]
        print(f"\n  {'against Section IV-A':<22}{'got':>9}{'paper':>9}{'':>8}")
        bad = 0
        for name, got, want in rows:
            ok = abs(got - want) <= TOL
            bad += not ok
            print(f"  {name:<22}{got:>9.2f}{want:>9.2f}{'  PASS' if ok else '  FAIL':>8}")
        if bad:
            print(f"\n  {bad} comparison(s) outside the {TOL}-point tolerance")
            return 1
        print(f"\n  all within {TOL} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
