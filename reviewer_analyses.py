#!/usr/bin/env python3
"""
reviewer_analyses.py — baselines, additive-model diagnostics, bootstrap CIs.

REVISION 04 (2026-10-23). Changes from the published version:

  1. BASELINE SET. `full_space_halving` replaces `race_only` as the
     no-screening comparison. It races all 103 configurations rather than a
     random shortlist, so c* is always in the field; a random shortlist
     rarely contains it, which made the old baseline a weak comparison. The
     old one is retained as `race_only_legacy` for continuity.

  2. UCB. The cached-repeat defect is fixed (see bl_ucb). Published UCB
     counted re-served instructions as independent observations: at budget
     960 every run inflated its statistics by 23 pulls on average, peaking
     at 99, without spending budget.

  3. DIAGNOSTICS. additive_diagnostics could not run against the current
     libra_select: main_effects returns three values since the measured-cell
     scoring change, and this unpacked two. It now unpacks three and passes
     the observed cells to predict(), matching what select() actually does.

  4. SEEDS. Every run, including the diagnostics, now uses
     rng = default_rng(10_000 * budget + seed_index). The diagnostics
     previously used default_rng(seed_index), a different schedule from the
     main curve, so the two were not comparable.

  5. TIES. np.argsort(..., kind="stable") throughout, so shortlist
     membership is a function of the seed rather than of the sort
     implementation. Unstable ordering changed the shortlist on 17.5% of
     runs.

  6. OUTPUT. Per-seed regrets and actual unique-query counts are written
     alongside the aggregate, so figures, per-target tables and paired tests
     all derive from one export.

Requires full_space_halving.py in the repository root and a libra_select.py
carrying the budget-cap and stable-tie fixes.

    python reviewer_analyses.py --boot 2000 --seeds 200
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from libra_select import (Replay, Hyper, select, bl_random, bl_race_only,
                          screen_sets, main_effects, predict, POOL, CACHE)
from full_space_halving import bl_successive_halving


def _seed(budget: int, index: int) -> np.random.Generator:
    """The one seed rule. Every method, target and analysis uses it."""
    return np.random.default_rng(10_000 * budget + index)


def _shortlist_size(hp: Hyper, budget: int) -> int:
    """Eq. (7), identical to select()'s own computation."""
    race_budget = int((1.0 - hp.screen_frac) * budget)
    return int(np.clip(2 ** int(np.floor(np.log2(
        max(race_budget / (2.0 * hp.n_race0), 4)))), 4, 32))


# ═══════════════════════════════════════════════════════════════════════════
# BASELINES
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

    next_instruction advances the shared cursor past any pair this arm has
    already been given. Without that, a wrapped cursor re-served instructions,
    replay.query returned the cached outcome for free, and n_pull counted it
    again -- tightening the confidence bound on evidence never paid for.
    Exhausted arms are masked so the search terminates.
    """
    replay.reset()
    cfgs = replay.configs
    pool = list(replay.instructions)
    rng.shuffle(pool)
    k = len(cfgs)
    n_pull = np.zeros(k)
    s_pull = np.zeros(k)
    cursor = 0
    seen_by_arm = [set() for _ in cfgs]

    def next_instruction(i):
        nonlocal cursor
        while True:
            x = pool[cursor % len(pool)]
            cursor += 1
            if x not in seen_by_arm[i]:
                seen_by_arm[i].add(x)
                return x

    # one pull per arm is already 103 queries; below that, pull at random
    order = rng.permutation(k)
    for i in order:
        if replay.n_queries >= budget:
            break
        s_pull[i] += replay.query(cfgs[i], next_instruction(i))
        n_pull[i] += 1

    while replay.n_queries < budget:
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(n_pull > 0, s_pull / np.maximum(n_pull, 1), 0.0)
            bonus = np.where(n_pull > 0,
                             np.sqrt(2 * np.log(max(replay.n_queries, 2))
                                     / np.maximum(n_pull, 1)),
                             1e9)
        scores = mean + bonus
        scores[n_pull >= len(pool)] = -np.inf
        if not np.isfinite(scores).any():
            break
        i = int(np.argmax(scores))
        s_pull[i] += replay.query(cfgs[i], next_instruction(i))
        n_pull[i] += 1

    mean = np.where(n_pull > 0, s_pull / np.maximum(n_pull, 1), -1.0)
    return cfgs[int(np.argmax(mean))], replay.n_queries


METHODS = {
    "LIBRA":              select,
    "full_space_halving": bl_successive_halving,
    "singles_sweep":      bl_singles_sweep,
    "ucb":                bl_ucb,
    "random":             bl_random,
    "race_only_legacy":   bl_race_only,
}


def run_baselines(df, budgets, seeds, hp, per_seed_out=None):
    rows, per_seed = [], []
    for target in POOL:
        rp = Replay(df, target)
        c_star, star = rp.optimum()
        for b in budgets:
            for name, fn in METHODS.items():
                regs, used = [], []
                for s in range(seeds):
                    pick, q = fn(rp, hp, b, _seed(b, s))
                    r = star - rp.true_asr(pick)
                    regs.append(r); used.append(q)
                    per_seed.append(dict(model=target, method=name, budget=b,
                                         seed=s, regret=float(r), pick=pick,
                                         queries=int(q),
                                         hit_optimum=bool(pick == c_star)))
                rows.append(dict(model=target, method=name, budget=b,
                                 regret=float(np.mean(regs)),
                                 regret_sd=float(np.std(regs)),
                                 queries_mean=float(np.mean(used)),
                                 queries_max=int(np.max(used)),
                                 over_budget=int(np.sum(np.array(used) > b)),
                                 c_star_asr=float(star), seeds=seeds))
    r = pd.DataFrame(rows)
    ps = pd.DataFrame(per_seed)
    if per_seed_out:
        ps.to_csv(per_seed_out, index=False)
        print(f"wrote {per_seed_out}  ({len(ps):,} rows)")

    print("=" * 78)
    print("MEAN REGRET (points) BY METHOD AND BUDGET")
    print("=" * 78)
    print((r.pivot_table(index="budget", columns="method",
                         values="regret") * 100).round(2).to_string())

    print("\nACTUAL UNIQUE QUERIES SPENT (max over targets and seeds)")
    print(r.pivot_table(index="budget", columns="method",
                        values="queries_max", aggfunc="max").to_string())
    viol = int(r.over_budget.sum())
    print(f"\nruns exceeding their nominal budget: {viol}"
          f"{'  <-- must be 0' if viol else '  (none)'}")

    b = max(budgets)
    print(f"\nPER TARGET at budget {b}  (Table 5)")
    t5 = (r[r.budget == b].pivot_table(index="model", columns="method",
                                       values="regret") * 100).round(2)
    print(t5.to_string())

    print(f"\nNORMALISED regret at budget {b}  (regret / c*)")
    n = r[(r.budget == b) & (r.method == "LIBRA")].copy()
    n["normalised_pct"] = n.regret / n.c_star_asr * 100
    print(n[["model", "c_star_asr", "regret", "normalised_pct"]]
          .assign(regret=lambda d: (d.regret * 100).round(2),
                  c_star_asr=lambda d: (d.c_star_asr * 100).round(1),
                  normalised_pct=lambda d: d.normalised_pct.round(1))
          .to_string(index=False))
    return r, ps


# ═══════════════════════════════════════════════════════════════════════════
# ADDITIVE MODEL DIAGNOSTICS   (Section V-E)
# ═══════════════════════════════════════════════════════════════════════════

def _screen_once(rp, hp, budget, rng):
    """
    One screening pass, identical to select()'s: the fixed design, the same
    instruction draw, the same estimates. Returns the raw components so each
    caller can score what it actually means to measure.

    Two different scores are wanted downstream and they must not be confused:

      predict(c, eff, direct)        PURE ADDITIVE. What the additive model
                                     alone predicts. This is the right score
                                     for the correlation against true cell
                                     performance, because probed cells must
                                     be scored by the model rather than by
                                     the measurement that would otherwise
                                     flatter it.

      predict(c, eff, direct, obs)   HYBRID. Measured value where the screen
                                     has one, additive otherwise. This is
                                     what select() ranks by, so it is the
                                     right score for shortlist membership.
    """
    pool = list(rp.instructions)
    rng.shuffle(pool)
    rp.reset()
    n_cells = hp.k_screen + 15
    n_screen = max(1, int(hp.screen_frac * budget) // n_cells)
    sc, si = screen_sets(rp.configs, hp.k_screen, rng)
    return main_effects(rp, sc, si, pool[:n_screen])


def additive_diagnostics(df, hp, budget, seeds):
    """
    Section IV argues composition is an interaction effect; Section V scores it
    additively. That is defensible only if the additive score ranks well enough
    to put c* on the shortlist, which the race then verifies. This measures
    exactly that, at one budget, under the main run's seed schedule.
    """
    print("\n" + "=" * 78)
    print(f"HOW WELL DOES THE ADDITIVE SCORE WORK?  (budget {budget}, "
          f"{seeds} seeds, seed rule 10000*budget+i)")
    print("=" * 78)
    k = _shortlist_size(hp, budget)
    print(f"shortlist size at this budget: k = {k}\n")
    print(f"{'target':<24} {'pearson':>8} {'spearman':>9} "
          f"{'c* shortlisted':>15} {'c* returned':>12} {'median rank':>12}")

    out = []
    for target in POOL:
        rp = Replay(df, target)
        c_star, star = rp.optimum()
        tied = {c for c in rp.configs if abs(rp.true_asr(c) - star) < 1e-12}
        cells = [c for c in rp.configs if c.count("|") == 2]
        truth = np.array([rp.true_asr(c) for c in cells])

        pear, spear, on_list, found, ranks = [], [], [], [], []
        for s in range(seeds):
            eff, direct, obs = _screen_once(rp, hp, budget, _seed(budget, s))
            # correlation: the ADDITIVE model's own prediction, no obs
            pred = np.array([predict(c, eff, direct) for c in cells])
            # shortlist: the hybrid score select() actually ranks by
            allp = np.array([predict(c, eff, direct, obs) for c in rp.configs])
            if pred.std() > 0 and truth.std() > 0:
                pear.append(float(np.corrcoef(pred, truth)[0, 1]))
                spear.append(float(pd.Series(pred).corr(pd.Series(truth),
                                                        method="spearman")))
            order = np.argsort(-allp, kind="stable")
            short = {rp.configs[i] for i in order[:k]}
            on_list.append(bool(short & tied))
            ranked = [rp.configs[i] for i in order]
            ranks.append(min(ranked.index(c) for c in tied) + 1)

            pick, _ = select(rp, hp, budget, _seed(budget, s))
            found.append(pick in tied)

        out.append(dict(model=target, pearson=np.mean(pear),
                        spearman=np.mean(spear), shortlisted=np.mean(on_list),
                        found=np.mean(found), median_rank=np.median(ranks)))
        print(f"{target:<24} {np.mean(pear):>8.3f} {np.mean(spear):>9.3f} "
              f"{np.mean(on_list):>14.1%} {np.mean(found):>11.1%} "
              f"{np.median(ranks):>12.1f}")

    d = pd.DataFrame(out)
    print(f"\nmean over targets: pearson {d.pearson.mean():.3f}, "
          f"spearman {d.spearman.mean():.3f}, "
          f"c* shortlisted {d.shortlisted.mean():.1%}, "
          f"c* returned {d.found.mean():.1%}")

    print("\nREGRET DECOMPOSITION  (is regret caused by the shortlist or the race?)")
    print(f"{'target':<24} {'total':>7} {'shortlist misses c*':>22} "
          f"{'shortlisted but lost':>22}")
    dec = []
    for target in POOL:
        rp = Replay(df, target)
        c_star, star = rp.optimum()
        tied = {c for c in rp.configs if abs(rp.true_asr(c) - star) < 1e-12}
        miss, lose, tot = [], [], []
        for s in range(seeds):
            pick, _ = select(rp, hp, budget, _seed(budget, s))
            r = star - rp.true_asr(pick)
            tot.append(r)
            eff, direct, obs = _screen_once(rp, hp, budget, _seed(budget, s))
            allp = np.array([predict(c, eff, direct, obs) for c in rp.configs])
            order = np.argsort(-allp, kind="stable")
            short = {rp.configs[i] for i in order[:k]}
            (lose if (short & tied) else miss).append(r)
        share_lost = (sum(lose) / sum(tot) * 100) if sum(tot) > 0 else 0.0
        dec.append(dict(model=target, total=np.mean(tot) * 100,
                        pct_regret_from_race=share_lost))
        print(f"{target:<24} {np.mean(tot)*100:>7.2f} "
              f"{(np.mean(miss)*100 if miss else 0):>14.2f} ({len(miss)/seeds:>4.0%}) "
              f"{(np.mean(lose)*100 if lose else 0):>14.2f} ({len(lose)/seeds:>4.0%})")
    print("\nshare of total regret arising on runs where c* WAS shortlisted:")
    for row in dec:
        print(f"  {row['model']:<24} {row['pct_regret_from_race']:>6.1f}%")
    return d


# ═══════════════════════════════════════════════════════════════════════════
# BOOTSTRAP CONFIDENCE INTERVALS
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
    print("\n" + "=" * 78)
    print(f"BOOTSTRAP 95% CIs OVER INSTRUCTIONS ({n_boot} resamples)")
    print("=" * 78)

    p = df[df.model.isin(POOL)]
    wide = p.pivot_table(index=["model", "instruction_id"],
                         columns="config", values="success")

    print("\nTRANSFER COST")
    per_model_star = {}
    for m in POOL:
        per_model_star[m] = wide.loc[m].mean().idxmax()
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
    ap.add_argument("--per-seed-out", default="results/per_seed_regrets.csv")
    ap.add_argument("--skip-bootstrap", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(CACHE)
    hp = Hyper()
    budgets = [int(b) for b in args.budgets.split(",")]

    r, ps = run_baselines(df, budgets, args.seeds, hp, args.per_seed_out)
    r.to_csv(args.out, index=False)
    additive_diagnostics(df, hp, args.diag_budget, args.seeds)
    if not args.skip_bootstrap:
        bootstrap(df, args.boot)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()