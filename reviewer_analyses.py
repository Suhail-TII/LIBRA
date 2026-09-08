#!/usr/bin/env python3
"""
reviewer_analyses.py — three analyses requested in review.

  #5  Two missing baselines:
        singles_sweep  spend the whole budget on the 15 single components
        ucb            a standard bandit over all 103 arms
      The first matters because on two targets the optimum IS a single
      component, so sweeping them finds it exactly and beats us on those folds.
      Reporting it is more honest than omitting it, and it locates our
      advantage precisely: we win where the optimum is composed.

  #6  Diagnostics for the additive scoring model, which Section IV argues is
      wrong in principle (composition interferes) yet Section V uses:
        - correlation between predicted and true ASR over the 88 cells
        - how often c* reaches the shortlist
        - regret decomposed into "c* not shortlisted" vs "shortlisted but lost"

  #7  Bootstrap confidence intervals on the transfer cost, the order effects,
      and the two results tables. Resampling is over INSTRUCTIONS, since that
      is the sampling unit: a different draw of 180 AdvBench instructions is
      the counterfactual a confidence interval should describe.

    python reviewer_analyses.py --boot 2000 --seeds 200
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from libra_select import (Replay, Hyper, select, bl_random, bl_race_only,
                          screen_sets, main_effects, predict, POOL, CACHE)


# ═══════════════════════════════════════════════════════════════════════════
# #5  MISSING BASELINES
# ═══════════════════════════════════════════════════════════════════════════

def bl_singles_sweep(replay: Replay, hp: Hyper, budget: int,
                     rng: np.random.Generator, **_):
    """
    Spend the entire budget on the 15 single components, then return the best.

    A strong baseline precisely where the optimum is a single component, which
    is the case on two of the five targets. It cannot ever return a composed
    cell, so it is bounded by the best single component and fails by exactly
    the synergy gap where the optimum is composed.
    """
    replay.reset()
    singles = [c for c in replay.configs if "|" not in c]
    pool = list(replay.instructions)
    rng.shuffle(pool)
    n = max(1, budget // len(singles))
    best, best_v = singles[0], -1.0
    for c in singles:
        if replay.n_queries + n > budget:
            break
        v = replay.batch(c, pool[:n])
        if v > best_v:
            best, best_v = c, v
    return best, replay.n_queries


def bl_ucb(replay: Replay, hp: Hyper, budget: int,
           rng: np.random.Generator, **_):
    """
    UCB1 over all 103 configurations as independent arms.

    The textbook bandit formulation of this problem, and the natural thing to
    reach for if the factorial structure is ignored. Each pull is one query on
    a fresh instruction; the arm with the highest empirical mean is returned.
    Its weakness here is that 103 arms is large relative to a few hundred
    pulls, so most arms are pulled once or twice.
    """
    replay.reset()
    cfgs = replay.configs
    pool = list(replay.instructions)
    rng.shuffle(pool)
    k = len(cfgs)
    n_pull = np.zeros(k)
    s_pull = np.zeros(k)
    cursor = 0

    def next_instruction():
        nonlocal cursor
        x = pool[cursor % len(pool)]
        cursor += 1
        return x

    # one pull per arm is already 103 queries; below that, pull at random
    order = rng.permutation(k)
    for i in order:
        if replay.n_queries >= budget:
            break
        s_pull[i] += replay.query(cfgs[i], next_instruction())
        n_pull[i] += 1

    while replay.n_queries < budget:
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(n_pull > 0, s_pull / np.maximum(n_pull, 1), 0.0)
            bonus = np.where(n_pull > 0,
                             np.sqrt(2 * np.log(max(replay.n_queries, 2))
                                     / np.maximum(n_pull, 1)),
                             1e9)
        i = int(np.argmax(mean + bonus))
        s_pull[i] += replay.query(cfgs[i], next_instruction())
        n_pull[i] += 1

    mean = np.where(n_pull > 0, s_pull / np.maximum(n_pull, 1), -1.0)
    return cfgs[int(np.argmax(mean))], replay.n_queries


def run_baselines(df, budgets, seeds, hp, seeds_out="results/regret_seeds.csv"):
    methods = {
        "LIBRA":         select,
        "singles_sweep": bl_singles_sweep,
        "ucb":           bl_ucb,
        "race_only":     bl_race_only,
        "random":        bl_random,
    }
    rows, per_seed = [], []
    for target in POOL:
        rp = Replay(df, target)
        _, star = rp.optimum()
        for b in budgets:
            for name, fn in methods.items():
                regs = []
                for s in range(seeds):
                    rng = np.random.default_rng(10_000 * b + s)
                    pick, _ = fn(rp, hp, b, rng)
                    reg = star - rp.true_asr(pick)
                    regs.append(reg)
                    per_seed.append(dict(model=target, method=name,
                                         budget=b, seed=s, regret=reg))
                rows.append(dict(model=target, method=name, budget=b,
                                 regret=float(np.mean(regs))))
    r = pd.DataFrame(rows)
    pd.DataFrame(per_seed).to_csv(seeds_out, index=False)
    print(f"  per-seed regret -> {seeds_out}")
 
    print("=" * 74)
    print("#5  MEAN REGRET (points) INCLUDING THE TWO NEW BASELINES")
    print("=" * 74)
    print((r.pivot_table(index="budget", columns="method",
                         values="regret") * 100).round(2).to_string())
 
    b = max(budgets)
    print(f"\nper target at budget {b}:")
    print((r[r.budget == b].pivot_table(index="model", columns="method",
                                        values="regret") * 100).round(2).to_string())
    return r


# ═══════════════════════════════════════════════════════════════════════════
# #6  ADDITIVE MODEL DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

def additive_diagnostics(df, hp, budget, seeds):
    """
    Section IV argues composition is an interaction effect; Section V scores it
    additively. That is defensible only if the additive score ranks well enough
    to put c* on the shortlist, which the race then verifies. This measures
    exactly that.
    """
    print("\n" + "=" * 74)
    print("#6  HOW WELL DOES THE ADDITIVE SCORE WORK?")
    print("=" * 74)
    print(f"{'target':<24} {'pearson':>8} {'spearman':>9} "
          f"{'c* shortlisted':>15} {'c* found':>9}")

    out = []
    for target in POOL:
        rp = Replay(df, target)
        c_star, star = rp.optimum()
        cells = [c for c in rp.configs if c.count("|") == 2]
        truth = np.array([rp.true_asr(c) for c in cells])

        pear, spear, on_list, found = [], [], [], []
        for s in range(seeds):
            rng = np.random.default_rng(s)
            n_cells = hp.k_screen + 15
            n_screen = max(1, int(hp.screen_frac * budget) // n_cells)
            pool = list(rp.instructions); rng.shuffle(pool)
            rp.reset()
            sc, si = screen_sets(rp.configs, hp.k_screen, rng)
            eff, direct = main_effects(rp, sc, si, pool[:n_screen])

            pred = np.array([predict(c, eff, direct) for c in cells])
            if pred.std() > 0 and truth.std() > 0:
                pear.append(float(np.corrcoef(pred, truth)[0, 1]))
                spear.append(float(pd.Series(pred).corr(pd.Series(truth),
                                                        method="spearman")))

            allp = np.array([predict(c, eff, direct) for c in rp.configs])
            race_budget = int((1 - hp.screen_frac) * budget)
            k = int(np.clip(2 ** int(np.floor(np.log2(
                max(race_budget / (2 * hp.n_race0), 4)))), 4, 32))
            short = [rp.configs[i] for i in np.argsort(-allp)[:k]]
            on_list.append(c_star in short)

            rng2 = np.random.default_rng(s)
            pick, _ = select(rp, hp, budget, rng2)
            found.append(pick == c_star)

        out.append(dict(model=target, pearson=np.mean(pear),
                        spearman=np.mean(spear),
                        shortlisted=np.mean(on_list), found=np.mean(found)))
        print(f"{target:<24} {np.mean(pear):>8.3f} {np.mean(spear):>9.3f} "
              f"{np.mean(on_list):>14.1%} {np.mean(found):>8.1%}")

    d = pd.DataFrame(out)
    print(f"\nmean over targets: pearson {d.pearson.mean():.3f}, "
          f"spearman {d.spearman.mean():.3f}, "
          f"c* shortlisted {d.shortlisted.mean():.1%}, "
          f"c* returned {d.found.mean():.1%}")

    # regret decomposition: is regret caused by a bad shortlist or a bad race?
    print("\nREGRET DECOMPOSITION at this budget")
    print(f"{'target':<24} {'total':>7} {'shortlist misses c*':>21} "
          f"{'race loses it':>14}")
    for target in POOL:
        rp = Replay(df, target)
        c_star, star = rp.optimum()
        miss, lose, tot = [], [], []
        for s in range(seeds):
            rng = np.random.default_rng(s)
            pick, _ = select(rp, hp, budget, rng)
            r = star - rp.true_asr(pick)
            tot.append(r)
            # rebuild the shortlist with the same seed to attribute the loss
            rng2 = np.random.default_rng(s)
            n_cells = hp.k_screen + 15
            n_screen = max(1, int(hp.screen_frac * budget) // n_cells)
            pool = list(rp.instructions); rng2.shuffle(pool)
            rp.reset()
            sc, si = screen_sets(rp.configs, hp.k_screen, rng2)
            eff, direct = main_effects(rp, sc, si, pool[:n_screen])
            allp = np.array([predict(c, eff, direct) for c in rp.configs])
            race_budget = int((1 - hp.screen_frac) * budget)
            k = int(np.clip(2 ** int(np.floor(np.log2(
                max(race_budget / (2 * hp.n_race0), 4)))), 4, 32))
            short = [rp.configs[i] for i in np.argsort(-allp)[:k]]
            (miss if c_star not in short else lose).append(r)
        print(f"{target:<24} {np.mean(tot)*100:>7.2f} "
              f"{np.mean(miss)*100 if miss else 0:>13.2f} ({len(miss)/seeds:>4.0%}) "
              f"{np.mean(lose)*100 if lose else 0:>8.2f} ({len(lose)/seeds:>4.0%})")
    return d


# ═══════════════════════════════════════════════════════════════════════════
# #7  BOOTSTRAP CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════════

def _boot_ci(vals, n_boot, rng, alpha=0.05):
    b = [np.mean(rng.choice(vals, size=len(vals), replace=True))
         for _ in range(n_boot)]
    return float(np.percentile(b, 100*alpha/2)), float(np.percentile(b, 100*(1-alpha/2)))


def bootstrap(df, n_boot):
    """
    Resample INSTRUCTIONS, not configurations or seeds. A confidence interval
    here should answer "what if we had drawn a different 180 AdvBench
    instructions", which is the sampling that actually happened.
    """
    rng = np.random.default_rng(0)
    print("\n" + "=" * 74)
    print(f"#7  BOOTSTRAP 95% CIs OVER INSTRUCTIONS ({n_boot} resamples)")
    print("=" * 74)

    p = df[df.model.isin(POOL)]
    wide = p.pivot_table(index=["model", "instruction_id"],
                         columns="config", values="success")

    # --- transfer cost -------------------------------------------------
    print("\nTRANSFER COST")
    per_model_star, per_model_peers = {}, {}
    for m in POOL:
        g = wide.loc[m]
        per_model_star[m] = g.mean().idxmax()
    boots = []
    for _ in range(n_boot):
        losses = []
        for m in POOL:
            g = wide.loc[m]
            idx = rng.choice(g.index, size=len(g), replace=True)
            gb = g.loc[idx]
            star = gb[per_model_star[m]].mean()
            peers = [gb[per_model_star[s]].mean() for s in POOL if s != m]
            losses.append(star - np.mean(peers))
        boots.append(np.mean(losses))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  mean transfer regret {np.mean(boots)*100:.1f} "
          f"[{lo*100:.1f}, {hi*100:.1f}]")

    # --- order effects -------------------------------------------------
    print("\nORDER EFFECT (framing-outer minus template-outer)")
    for m in POOL:
        g = wide.loc[m]
        fo = [c for c in g.columns if c.endswith("libra_then_utos")]
        to = [c for c in g.columns if c.endswith("utos_then_libra")]
        b = []
        for _ in range(n_boot):
            idx = rng.choice(g.index, size=len(g), replace=True)
            gb = g.loc[idx]
            b.append(gb[fo].values.mean() - gb[to].values.mean())
        lo, hi = np.percentile(b, [2.5, 97.5])
        sig = "" if lo <= 0 <= hi else "  *"
        print(f"  {m:<24} {np.mean(b)*100:>+6.1f} "
              f"[{lo*100:>+6.1f}, {hi*100:>+6.1f}]{sig}")
    print("  * interval excludes zero")

    # --- c* ASR --------------------------------------------------------
    print("\nc* ASR")
    for m in POOL:
        g = wide.loc[m]
        col = per_model_star[m]
        b = [g.loc[rng.choice(g.index, size=len(g), replace=True), col].mean()
             for _ in range(n_boot)]
        lo, hi = np.percentile(b, [2.5, 97.5])
        print(f"  {m:<24} {np.mean(b)*100:>5.1f}% "
              f"[{lo*100:.1f}, {hi*100:.1f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="120,240,360,480,720,960")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--diag-budget", type=int, default=960)
    ap.add_argument("--out", default="results/reviewer_analyses.csv")
    args = ap.parse_args()

    df = pd.read_parquet(CACHE)
    hp = Hyper()
    budgets = [int(b) for b in args.budgets.split(",")]

    r = run_baselines(df, budgets, args.seeds, hp)
    r.to_csv(args.out, index=False)
    additive_diagnostics(df, hp, args.diag_budget, min(args.seeds, 100))
    bootstrap(df, args.boot)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()