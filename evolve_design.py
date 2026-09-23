#!/usr/bin/env python3
"""
evolve_design.py — evolve the screening design against measured regret, and
validate it on a target the evolution never saw.

WHY
---
The screening subset is currently chosen by a structural criterion: maximise
how many distinct (framing, template) and (framing, order) combinations the 22
cells contain. That criterion is defensible but we have evidence it does not
bind -- a second design built on a different criterion (count balance) performs
identically. So the question this script answers is whether a better design
exists at all, judged by the quantity we actually care about, which is regret.

Optimising a design against regret on the targets we then report would be
circular, and is the same objection the reviewer raised about how the eleven
framings were selected. We therefore evolve on four targets and measure on the
fifth, rotating across all five. A design evolved without ever seeing a target,
which still performs well on that target, has found something structural rather
than memorised the models it was tuned on. If evolved designs only work on the
targets they saw, that is also a result, and it argues for keeping the
coverage design.

WHAT THE SETTINGS MEAN
----------------------
population   How many candidate designs exist at once. Larger explores more of
             the space but costs proportionally more. 30 is enough diversity
             for a space this structured.

generations  How many rounds of breed-score-select to run. GAs on small spaces
             converge early; we run 25 and report the fitness curve so it is
             visible whether it plateaued.

elite        How many of the best designs survive each generation untouched.
             Without this, a good design can be lost to an unlucky crossover.

tournament   How a parent is chosen: sample `tournament` designs at random and
             take the best. Larger values favour the already-good more
             strongly, which converges faster but can get stuck.

mutation     Probability that a framing's cell pair is perturbed after
             crossover. This is what keeps new combinations entering the
             population.

seeds        How many instruction draws each design is scored on. A design's
             regret varies by the draw, so this is a sample size. 30 is enough
             to rank designs; the winner is re-scored at 200 before being
             reported, because a design can win a generation by luck.

budget       Which query budget fitness is measured at. We use 240: the
             screening-size sweep shows regret spanning 6.32 to 9.00 there,
             where at 960 it spans 1.15 to 1.81, so 240 is where the design
             choice has room to matter.

    python evolve_design.py --generations 25 --population 30
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import zlib

from libra_select import Replay, Hyper, select, POOL, CACHE
from attacks.libra_framings import UTOS_TOP4, STAGE_B_FRAMINGS

ORDERS = ["libra_then_utos", "utos_then_libra"]
OUT = Path("results/evolved_design")


# ── representation ─────────────────────────────────────────────────────────
# A design is a dict: framing -> the two (template, order) pairs it uses.
# Crossover can then take whole framings from either parent and the result is
# always valid, because the two-cells-per-framing constraint is structural
# rather than something the fitness function has to enforce.

def random_design(rng, framings, options):
    return {f: list(rng.choice(len(options), size=2, replace=False))
            for f in framings}


def to_cells(design, options):
    return [f"{f}|{options[i][0]}|{options[i][1]}"
            for f, idx in design.items() for i in idx]


def crossover(a, b, rng, framings):
    """Each framing's pair comes from one parent or the other."""
    return {f: list(a[f] if rng.random() < 0.5 else b[f]) for f in framings}


def mutate(design, rng, options, rate):
    """Swap one cell of a framing for a different (template, order) pair."""
    d = {f: list(v) for f, v in design.items()}
    for f in d:
        if rng.random() < rate:
            keep = d[f][rng.integers(2)]
            choices = [i for i in range(len(options)) if i != keep]
            d[f] = [keep, int(rng.choice(choices))]
    return d


# ── fitness ────────────────────────────────────────────────────────────────

# def fitness(cells, replays, hp, budget, seeds):
#     """
#     Mean regret across the training targets. Lower is better.

#     The design enters through hp_cells, which libra_select's screen_sets reads
#     when present, so nothing else about the procedure changes.
#     """
#     regs = []
#     for rp, star in replays:
#         for s in range(seeds):
#             rng = np.random.default_rng(s)
#             pick, _ = select(rp, hp, budget, rng, screen_cells=cells)
#             regs.append(star - rp.true_asr(pick))
#     return float(np.mean(regs)) * 100

def fitness(cells, replays, hp, budget, seeds, lam=0.0):
    """
    Mean regret across the training targets, optionally penalised for lost
    coverage.

    Unpenalised, the GA drifts away from structural balance: evolved designs
    reached 19-21 of 44 framing-template pairs against the fixed design's 22,
    paying in coverage for fit to whichever targets it happened to see. The
    penalty prices that drift. lam is in points of regret per missing pair, so
    lam = 0.5 says one lost pair is worth half a point.
    """
    regs = []
    for rp, star in replays:
        for s in range(seeds):
            rng = np.random.default_rng(s)
            pick, _ = select(rp, hp, budget, rng, screen_cells=cells)
            regs.append(star - rp.true_asr(pick))
    base = float(np.mean(regs)) * 100
    if lam == 0.0:
        return base
    ft, fo = coverage(cells)
    return base + lam * ((22 - ft) + (22 - fo))


def coverage(cells):
    ft = {(c.split("|")[0], c.split("|")[1]) for c in cells}
    fo = {(c.split("|")[0], c.split("|")[2]) for c in cells}
    return len(ft), len(fo)


# ── evolution ──────────────────────────────────────────────────────────────

def evolve(replays, hp, args, framings, options, rng, log=None):
    pop = [random_design(rng, framings, options) for _ in range(args.population)]
    best, best_fit = None, 1e9
    for gen in range(args.generations):
        scored = sorted(
            # ((fitness(to_cells(d, options), replays, hp, args.budget,
            #           args.seeds), d) for d in pop),
            ((fitness(to_cells(d, options), replays, hp, args.budget, args.seeds, args.lam), d) for d in pop),
            key=lambda t: t[0])
        if scored[0][0] < best_fit:
            best_fit, best = scored[0][0], scored[0][1]
        if log is not None:
            log.append(dict(generation=gen, best=scored[0][0],
                            median=scored[len(scored)//2][0]))
        # elitism, then tournament-selected breeding
        nxt = [d for _, d in scored[:args.elite]]
        while len(nxt) < args.population:
            def pick():
                cand = [scored[i] for i in
                        rng.choice(len(scored), size=args.tournament,
                                   replace=False)]
                return min(cand, key=lambda t: t[0])[1]
            child = crossover(pick(), pick(), rng, framings)
            nxt.append(mutate(child, rng, options, args.mutation))
        pop = nxt
    return best, best_fit

def evolve_final(df, hp, args, framings, options):
    """
    Evolve one design on ALL five targets, for deployment.

    The leave-one-out experiment validates the PROCEDURE: a design evolved
    without seeing a target still improves on it, so the gain is structural
    rather than memorised. Having established that, the design that ships is
    trained on everything available, as is standard once cross-validation has
    justified the method. Reporting this design's regret on the same five
    targets would be circular; the numbers that justify it are the held-out
    ones in leave_one_out.csv.
    """
    replays = []
    for m in POOL:
        rp = Replay(df, m)
        replays.append((rp, rp.optimum()[1]))

    rng = np.random.default_rng(12345)
    log = []
    best, fit = evolve(replays, hp, args, framings, options, rng, log)
    cells = sorted(to_cells(best, options))
    ft, fo = coverage(cells)

    print("\n" + "=" * 74)
    print(f"FINAL DESIGN  (all 5 targets, lambda={args.lam}, "
          f"{args.generations} generations)")
    print("=" * 74)
    print(f"fitness (penalised)     {fit:.2f}")
    print(f"coverage                {ft}/44 framing-template, {fo}/22 framing-order")
    print("\nSCREEN_DESIGN = [")
    for c in cells:
        print(f'    "{c}",')
    print("]")

    with open(OUT / "final_design.json", "w") as fh:
        json.dump({"lam": args.lam, "generations": args.generations,
                   "population": args.population, "seeds": args.seeds,
                   "budget": args.budget, "fitness": fit,
                   "ft_pairs": ft, "fo_pairs": fo,
                   "cells": cells}, fh, indent=2)
    pd.DataFrame(log).to_csv(OUT / "final_fitness_curve.csv", index=False)
    print(f"\nwrote {OUT}/final_design.json")
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", type=int, default=30)
    ap.add_argument("--generations", type=int, default=25)
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--tournament", type=int, default=3)
    ap.add_argument("--mutation", type=float, default=0.25)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--final-seeds", type=int, default=200)
    ap.add_argument("--lam", type=float, default=0.5,
                    help="regret points charged per missing coverage pair")
    ap.add_argument("--final", action="store_true",
                    help="also evolve one design on all five targets")
    ap.add_argument("--budget", type=int, default=240)
    ap.add_argument("--eval-budgets", default="240,960")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(CACHE)
    hp = Hyper()
    framings = sorted(STAGE_B_FRAMINGS)
    options = [(t, o) for t in sorted(UTOS_TOP4) for o in ORDERS]
    eval_budgets = [int(b) for b in args.eval_budgets.split(",")]

    # the current design, for comparison
    from libra_select import SCREEN_DESIGN as CURRENT

    rows, curves = [], []
    for held in POOL:
        train = [m for m in POOL if m != held]
        replays = []
        for m in train:
            rp = Replay(df, m)
            replays.append((rp, rp.optimum()[1]))

        # Python randomises str hashing per process, so hash(held) would give
        # a different seed on every invocation and the experiment would not
        # reproduce. crc32 is stable across runs.
        rng = np.random.default_rng(zlib.crc32(held.encode()))
        log = []
        best, train_fit = evolve(replays, hp, args, framings, options, rng, log)
        cells = to_cells(best, options)
        curves += [dict(held_out=held, **r) for r in log]

        rp_h = Replay(df, held)
        star_h = rp_h.optimum()[1]

        def regret(design_cells, budget, seeds):
            r = []
            for s in range(seeds):
                pick, _ = select(rp_h, hp, budget, np.random.default_rng(s),
                                 screen_cells=design_cells)
                r.append(star_h - rp_h.true_asr(pick))
            return float(np.mean(r)) * 100

        ft, fo = coverage(cells)
        row = dict(held_out=held, train_regret=train_fit,
                   ft_pairs=ft, fo_pairs=fo)
        for b in eval_budgets:
            row[f"evolved@{b}"] = regret(cells, b, args.final_seeds)
            row[f"current@{b}"] = regret(CURRENT, b, args.final_seeds)
        rows.append(row)

        print(f"\nheld out {held}")
        print(f"  train regret (4 targets, {args.seeds} seeds) {train_fit:.2f}")
        print(f"  coverage: {ft}/44 framing-template, {fo}/22 framing-order")
        for b in eval_budgets:
            e, c = row[f"evolved@{b}"], row[f"current@{b}"]
            flag = "evolved better" if e < c else "current better"
            print(f"  on held-out target @ {b:>4}q: evolved {e:>6.2f}   "
                  f"current {c:>6.2f}   ({flag})")
        with open(OUT / f"design_{held.replace('.', '_')}.json", "w") as fh:
            json.dump({"held_out": held, "cells": sorted(cells),
                       "train_regret": train_fit}, fh, indent=2)

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "leave_one_out.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "fitness_curves.csv", index=False)

    print("\n" + "=" * 74)
    print("LEAVE-ONE-TARGET-OUT SUMMARY")
    print("=" * 74)
    print(r.round(2).to_string(index=False))
    for b in eval_budgets:
        e, c = r[f"evolved@{b}"].mean(), r[f"current@{b}"].mean()
        wins = int((r[f"evolved@{b}"] < r[f"current@{b}"]).sum())
        print(f"\n@ {b} queries: evolved {e:.2f}  current {c:.2f}  "
              f"evolved wins on {wins}/{len(r)} held-out targets")
    print(f"\nwrote {OUT}/")
    print("\nA design evolved without seeing a target should still beat the "
          "current one\non that target if the gain is structural. If it only "
          "wins on the four it\nsaw, it has fitted to them and the coverage "
          "design should be kept.")

    if args.final:
        evolve_final(df, hp, args, framings, options)

if __name__ == "__main__":
    main()