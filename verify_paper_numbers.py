#!/usr/bin/env python3
"""
verify_paper_numbers.py — regenerate every figure quoted in the paper from the
released caches, grouped by the section that cites it.

Numbers have moved twice: once when the judge cache was deduplicated, and again
when the screening subset was fixed and probed cells began using their measured
values. This prints the current value of everything, so the draft can be
checked against one source rather than against notes of different vintages.

    python verify_paper_numbers.py                 # everything
    python verify_paper_numbers.py --sections 4 5  # just those
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

POOL = ["Llama-2-7B-Chat", "Llama-3.1-8B-Instruct", "Qwen3-14B",
        "gpt-5.1", "Ministral-3-8B-Instruct"]
SHORT = {"Llama-2-7B-Chat": "Llama-2-7B",
         "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
         "Qwen3-14B": "Qwen3-14B",
         "gpt-5.1": "GPT-5.1",
         "Ministral-3-8B-Instruct": "Ministral-3-8B"}
FLOOR = ["gpt-5-mini", "gpt-5-nano"]

CACHE = "results/replay_cache.parquet"
JUDGE = "results/judge_cache.parquet"
EVALH = "results/eval_half_cache.parquet"
LOMO = "results/lomo_results.csv"
RANAL = "results/reviewer_analyses.csv"


def head(n, title):
    print("\n" + "=" * 76)
    print(f"SECTION {n}  {title}")
    print("=" * 76)


# ═══════════════════════════════════════════════════════════════════════════
def section3(d):
    head("III", "Experimental Setup")

    p = d[d.model.isin(POOL)]
    print(f"configurations per target : {sorted(p.groupby('model').config.nunique().unique())}")
    print("instructions per target   :")
    for m, n in p.groupby("model").instruction_id.nunique().items():
        print(f"    {SHORT[m]:<16} {n}")

    n_cfg = p.config.nunique()
    print(f"\ndesign space              : {n_cfg} configurations")
    cells = [c for c in p.config.unique() if c.count("|") == 2]
    print(f"    composed cells        : {len(cells)}")
    print(f"    single components     : {n_cfg - len(cells)}")
    print(f"exhaustive cost per target: {n_cfg * 180:,} queries")
    print(f"pairs in the design space : {len(p):,}   "
          f"(= {n_cfg} x (4 x 180 + 178))")

    # the margin claim: where c* is a single component, how far ahead is it
    print("\nSINGLE-COMPONENT OPTIMA (the 'declining to compose' margin)")
    a = p.groupby(["model", "config"]).success.mean().unstack()
    for m in POOL:
        v = a.loc[m]
        star = v.idxmax()
        if "|" in star:
            continue
        best_cell = v[[c for c in v.index if c.count("|") == 2]].max()
        print(f"    {SHORT[m]:<16} {v.max():.1%} alone vs {best_cell:.1%} best cell"
              f"   margin +{(v.max()-best_cell)*100:.1f} points")

    # judge cache volume, for the reconciliation sentence
    if Path(JUDGE).exists():
        j = pd.read_parquet(JUDGE)
        print(f"\nscored responses (all stages): {len(j):,}")
        print(f"    reconciles as 143 x (4 x 180 + 178) = "
              f"{143 * (4*180 + 178):,}")


# ═══════════════════════════════════════════════════════════════════════════
def section4(d):
    head("IV", "Design Space Analysis")
    p = d[d.model.isin(POOL)]
    a = p.groupby(["model", "config"]).success.mean().unstack()

    print("TABLE 2  c* and transfer cost")
    print(f"    {'target':<16} {'c* ASR':>7} {'best tr':>8} {'mean tr':>8} {'worst tr':>9}")
    losses = []
    for m in POOL:
        star = a.loc[m].max()
        peers = [a.loc[m, a.loc[s].idxmax()] for s in POOL if s != m]
        losses.append(star - np.mean(peers))
        print(f"    {SHORT[m]:<16} {star:>6.1%} {max(peers):>7.1%} "
              f"{np.mean(peers):>7.1%} {min(peers):>8.1%}")
    print(f"\n    MEAN TRANSFER REGRET  {np.mean(losses)*100:.1f} points")
    print(f"    distinct c*           {a.loc[POOL].idxmax(axis=1).nunique()}/{len(POOL)}")

    # cross-target rank correlation over the whole space (the Fig. 3 number)
    corr = a.loc[POOL].T.corr(method="spearman").values
    off = corr[np.triu_indices_from(corr, k=1)]
    print(f"\n    pairwise rank correlation between targets, all {a.shape[1]} configs:")
    print(f"      {off.min():+.2f} to {off.max():+.2f}, mean {off.mean():+.2f}")

    print("\nTABLE 3  synergy")
    print(f"    {'target':<16} {'best comp':>10} {'best cell':>10} {'synergy':>8} {'cells>':>8}")
    rows = []
    for m in POOL + [f for f in FLOOR if f in d.model.unique()]:
        v = d[d.model == m].groupby("config").success.mean()
        single = v[[c for c in v.index if "|" not in c]]
        cells = v[[c for c in v.index if c.count("|") == 2]]
        n_better = int((cells > single.max()).sum())
        rows.append((m, single.max(), cells.max(), cells.max() - single.max()))
        name = SHORT.get(m, m)
        print(f"    {name:<16} {single.max():>9.1%} {cells.max():>9.1%} "
              f"{(cells.max()-single.max())*100:>+7.1f} {n_better:>5}/88")
    pool_rows = [r for r in rows if r[0] in POOL]
    rho = pd.Series([r[1] for r in pool_rows]).corr(
        pd.Series([r[3] for r in pool_rows]), method="spearman")
    print(f"\n    Spearman(best component, synergy) = {rho:+.3f}  (n=5)")

    tot = better = 0
    for m in POOL:
        v = a.loc[m]
        for c in [x for x in v.index if x.count("|") == 2]:
            f, t, _ = c.split("|")
            tot += 1
            better += v[c] > max(v.get(f, 0), v.get(t, 0))
    print(f"    cells beating BOTH components: {better}/{tot} = {better/tot:.1%}")
    print(f"    -> interference rate {1-better/tot:.1%}")

    print("\nTABLE 4  composition order")
    c = p[p.config.str.count(r"\|") == 2].copy()
    c["order"] = c.config.str.split("|").str[2]
    o = c.groupby(["model", "order"]).success.mean().unstack() * 100
    o["delta"] = o["libra_then_utos"] - o["utos_then_libra"]
    for m in POOL:
        r = o.loc[m]
        print(f"    {SHORT[m]:<16} {r['libra_then_utos']:>6.1f} "
              f"{r['utos_then_libra']:>6.1f} {r['delta']:>+7.1f}")

    print("\nIV-D  evaluator")
    if not Path(JUDGE).exists():
        print("    judge_cache.parquet missing")
    else:
        j = pd.read_parquet(JUDGE)
        lg = j.jailbreak_success_llama_guard.fillna(False).astype(bool)
        hb = j.jailbreak_success_harmbench.fillna(False).astype(bool)
        dis = lg != hb
        print(f"    rows                  {len(j):,}")
        print(f"    disagreement          {dis.mean():.1%}")
        print(f"    one-directional       {(lg & ~hb).sum()/dis.sum():.1%}")
        if "harmbench_confidence" in j:
            print(f"    HB median conf        {j.loc[dis,'harmbench_confidence'].median():.4f}")
        if "hit_token_cap" in j:
            cap = j.hit_token_cap.fillna(False).astype(bool)
            print(f"    truncated | disagree  {cap[dis].mean():.1%}")
            print(f"    truncated | agree     {cap[~dis].mean():.1%}")
        print(f"    strict                {(lg & hb).mean():.1%}")
        print(f"    lenient               {(lg | hb).mean():.1%}")
        if "jailbreak_success_ensemble" in j:
            ens = j.jailbreak_success_ensemble.fillna(False).astype(bool)
            print(f"    2-of-3 ensemble       {ens.mean():.1%}")
            print(f"    heuristic-decided     {(ens & ~(lg & hb)).sum()/ens.sum():.1%}")

    print("\n    floor cases")
    for m in FLOOR:
        g = d[d.model == m]
        if g.empty:
            continue
        v = g.groupby("config").success.mean()
        n = g.instruction_id.nunique()
        print(f"      {m:<12} best {v.max():>5.1%} ({int(round(v.max()*n))}/{n})"
              f"   configs > 0: {(v > 0).sum()}/103")


# ═══════════════════════════════════════════════════════════════════════════
def section5():
    head("V", "The LIBRA Search Procedure")

    if Path(RANAL).exists():
        r = pd.read_csv(RANAL)
        print("FIGURE 3  mean regret by budget and method")
        piv = r.pivot_table(index="budget", columns="method", values="regret") * 100
        print(piv.round(2).to_string())
        b = int(piv.index.max())
        print(f"\nTABLE 5  per-target at {b} queries")
        t = r[r.budget == b].pivot_table(index="model", columns="method",
                                         values="regret") * 100
        print(t.round(2).to_string())
    elif Path(LOMO).exists():
        r = pd.read_csv(LOMO)
        print("(reviewer_analyses.csv missing; using lomo_results.csv)")
        piv = r.pivot_table(index="budget", columns="method",
                            values="regret_mean") * 100
        print(piv.round(2).to_string())
    else:
        print("no results file; run libra_select.py --lomo and reviewer_analyses.py")
        return

    if Path(LOMO).exists():
        l = pd.read_csv(LOMO)
        lib = l[l.method == "LIBRA"]
        b = int(lib.budget.max())
        print(f"\nNORMALISED regret at {b} queries (regret / c*)")
        x = lib[lib.budget == b]
        for _, row in x.iterrows():
            print(f"    {SHORT.get(row.model, row.model):<16} "
                  f"c* {row.asr_star:>6.1%}  regret {row.regret_mean*100:>5.2f}  "
                  f"norm {row.regret_mean/row.asr_star:>6.1%}")
        print(f"\nSPREAD across seeds (the distribution, not the mean)")
        for b2 in sorted(lib.budget.unique()):
            y = lib[lib.budget == b2]
            print(f"    budget {b2:>4}: sd {y.regret_sd.mean()*100:>5.2f}  "
                  f"p90 {y.regret_p90.mean()*100:>5.2f}  "
                  f"hit-optimum {y.hit_optimum.mean():>5.1%}")


def coverage_design(per_framing, seed=0, trials=200_000):
    """
    Build a coverage design with `per_framing` cells per framing.

    Constructed fresh at each size rather than truncating the 22-cell design:
    screen_sets returns a fixed 22 cells, so asking it for 33 or 44 silently
    returns 22 and the sweep compares nothing.
    """
    from collections import Counter
    from attacks.libra_framings import UTOS_TOP4, STAGE_B_FRAMINGS
    F, T = sorted(STAGE_B_FRAMINGS), sorted(UTOS_TOP4)
    orders = ["libra_then_utos", "utos_then_libra"]
    opts = [(t, o) for t in T for o in orders]
    rng = np.random.default_rng(seed)
    best, best_s = None, (-1, -1, -1e9)
    for _ in range(trials):
        d = [(f, *opts[i]) for f in F
             for i in rng.choice(len(opts), size=per_framing, replace=False)]
        ft = len({(c[0], c[1]) for c in d})
        fo = len({(c[0], c[2]) for c in d})
        tc, oc = Counter(c[1] for c in d), Counter(c[2] for c in d)
        bal = -(np.std([tc[x] for x in T]) + np.std([oc[x] for x in orders]))
        if (ft, fo, bal) > best_s:
            best, best_s = d, (ft, fo, bal)
    return [f"{a}|{b}|{c}" for a, b, c in best], best_s


def sweep(d, seeds=200):
    """Table 6: screening size against budget."""
    from libra_select import Replay, Hyper, select
    head("V-B", "Screening size sweep (Table 6)")

    designs = {}
    for n in (1, 2, 3, 4):
        cells, s = coverage_design(n)
        designs[len(cells)] = cells
        print(f"    {len(cells):>2} cells: coverage {s[0]}/44 framing-template, "
              f"{s[1]}/22 framing-order")

    print(f"\n{'budget':>7} " + " ".join(f"{k:>8}" for k in sorted(designs)))
    for B in (120, 240, 360, 480, 720, 960, 1440):
        row = []
        for k in sorted(designs):
            # k_screen must match the design size or n_s is computed for the
            # wrong number of cells and the screening budget is overspent.
            hp = Hyper(k_screen=k)
            regs = []
            for m in POOL:
                rp = Replay(d, m); _, star = rp.optimum()
                regs += [star - rp.true_asr(
                    select(rp, hp, B, np.random.default_rng(s),
                           screen_cells=designs[k])[0]) for s in range(seeds)]
            row.append(np.mean(regs) * 100)
        mark = int(np.argmin(row))
        cells = [f"{x:>8.2f}" + ("*" if i == mark else " ")
                 for i, x in enumerate(row)]
        print(f"{B:>7} " + " ".join(cells))
    print("\n    * best in row")


# ═══════════════════════════════════════════════════════════════════════════
def section6(d):
    head("VI", "Discussion")
    p = d[d.model.isin(POOL)]
    a = p.groupby(["model", "config"]).success.mean().unstack()

    print("SPARSE-LANDSCAPE CLAIM (why Llama-2 and GPT-5.1 are hard)")
    for m in POOL:
        v = a.loc[m]
        print(f"    {SHORT[m]:<16} mean ASR across all 103 {v.mean():>6.1%}   "
              f"c* {v.max():>6.1%}")

    print("\nPROBE vs HELD-OUT EVALUATION (Section III-B claim)")
    if not Path(EVALH).exists():
        print("    eval_half_cache.parquet missing")
        return
    e = pd.read_parquet(EVALH)
    key = "cell" if "cell" in e.columns else "config"
    ev = e.groupby(["model", key]).success_strict.mean()
    forf, rhos, drops = [], [], []
    for m in sorted(e.model.unique()):
        got = [c for c in ev.loc[m].index]
        pv = np.array([a.loc[m, c] for c in got if c in a.columns])
        evv = np.array([ev.loc[m, c] for c in got if c in a.columns])
        got = [c for c in got if c in a.columns]
        sp = got[int(np.argmax(pv))]
        se = got[int(np.argmax(evv))]
        f = ev.loc[m, se] - ev.loc[m, sp]
        rho = pd.Series(pv).corr(pd.Series(evv), method="spearman")
        drop = a.loc[m, sp] - ev.loc[m, sp]
        forf.append(f); rhos.append(rho); drops.append(drop)
        print(f"    {SHORT.get(m,m):<16} probe {a.loc[m,sp]:>6.1%} "
              f"eval {ev.loc[m,sp]:>6.1%}  forfeit {f*100:>4.1f}  rho {rho:>+.2f}"
              f"   {'same c*' if sp == se else 'different c* on eval'}")
    print(f"\n    mean forfeit {np.mean(forf)*100:.1f} points | "
          f"mean rho {np.mean(rhos):+.2f} | "
          f"mean ASR change probe->eval {-np.mean(drops)*100:+.1f} points")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", nargs="*", default=["3", "4", "5", "6"])
    ap.add_argument("--sweep", action="store_true",
                help="also regenerate Table 6 (slow: builds designs)")
    args = ap.parse_args()

    d = pd.read_parquet(CACHE)
    if "3" in args.sections: section3(d)
    if "4" in args.sections: section4(d)
    if "5" in args.sections: section5()
    if "6" in args.sections: section6(d)
    if args.sweep: sweep(d)
    print()


if __name__ == "__main__":
    main()
