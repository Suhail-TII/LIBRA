#!/usr/bin/env python3
"""
verify_section4.py — regenerate every number quoted in Section IV from the
replay cache, so the section is internally consistent and reproducible.

Some figures in the draft were computed at different points in the project on
different subsets. This script recomputes all of them from one source, over the
five evaluated targets only. Any mismatch against the draft is the draft's
error, not this script's.

    python verify_section4.py
"""

import glob
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

POOL = ["Llama-2-7B-Chat", "Llama-3.1-8B-Instruct", "Qwen3-14B",
        "gpt-5.1", "Ministral-3-8B-Instruct"]
FLOOR = ["gpt-5-mini", "gpt-5-nano"]

d = pd.read_parquet("results/replay_cache.parquet")
p = d[d.model.isin(POOL)]
asr = p.groupby(["model", "config"]).success.mean().unstack()

print("=" * 74)
print("HEADER: total measurement volume")
print("=" * 74)
print(f"prompt-response pairs (5 targets) : {len(p):,}")
print(f"configurations per target         : {p.groupby('model').config.nunique().unique()}")
print(f"instructions per target           :")
print(p.groupby("model").instruction_id.nunique().to_string())

# ── IV-A ────────────────────────────────────────────────────────────────
print("\n" + "=" * 74)
print("IV-A  c* AND TRANSFER COST")
print("=" * 74)
rows = []
for t in POOL:
    star_cfg, star = asr.loc[t].idxmax(), asr.loc[t].max()
    peers = [asr.loc[t, asr.loc[s].idxmax()] for s in POOL if s != t]
    rows.append(dict(model=t, cstar=star_cfg, star=star,
                     best_tr=max(peers), worst_tr=min(peers),
                     mean_tr=float(np.mean(peers))))
r = pd.DataFrame(rows)
for x in r.itertuples():
    kind = "framing only" if x.cstar.count("|") == 0 else "composed"
    print(f"{x.model:<24} {x.star:>6.1%}  [{kind}] {x.cstar}")
    print(f"{'':<24} transferred: best {x.best_tr:>5.1%}  "
          f"mean {x.mean_tr:>5.1%}  worst {x.worst_tr:>5.1%}")
print(f"\nMEAN TRANSFER REGRET: {(r.star - r.mean_tr).mean():.1%}")
print(f"distinct c* configurations: {r.cstar.nunique()}/{len(POOL)}")
n_framing_only = sum("|" not in c for c in r.cstar)
print(f"c* is framing-only on      : {n_framing_only}/{len(POOL)}")
# ── IV-B ────────────────────────────────────────────────────────────────
print("\n" + "=" * 74)
print("IV-B  SYNERGY")
print("=" * 74)
syn = []
for t in POOL:
    v = asr.loc[t]
    single = v[[c for c in v.index if "|" not in c]]
    cells = v[[c for c in v.index if c.count("|") == 2]]
    n_better = int((cells > single.max()).sum())
    syn.append(dict(model=t, comp=single.max(), cell=cells.max(),
                    synergy=cells.max() - single.max(), n=n_better))
s = pd.DataFrame(syn).sort_values("synergy", ascending=False)
for x in s.itertuples():
    print(f"{x.model:<24} component {x.comp:>6.1%}  cell {x.cell:>6.1%}  "
          f"synergy {x.synergy*100:>+6.1f}  cells> {x.n:>2}/88")
rho = s.comp.corr(s.synergy, method="spearman")
print(f"\nSpearman(best component, synergy) = {rho:+.3f}")

# fraction of cells beating BOTH their components
tot_better = tot = 0
for t in POOL:
    v = asr.loc[t]
    for c in [c for c in v.index if c.count("|") == 2]:
        f, tp, _ = c.split("|")
        base = max(v.get(f, 0), v.get(tp, 0))
        tot += 1
        tot_better += v[c] > base
print(f"cells beating BOTH components: {tot_better}/{tot} = {tot_better/tot:.1%}")
print(f"-> interference rate: {1 - tot_better/tot:.1%}")

print("\ninterference within Llama-3.1 (framing alone vs its best cell):")
v = asr.loc["Llama-3.1-8B-Instruct"]
fr = sorted({c.split("|")[0] for c in v.index if c.count("|") == 2})
for f in fr:
    alone = v.get(f, np.nan)
    best = v[[c for c in v.index if c.startswith(f + "|")]].max()
    print(f"  {f:<26} {alone:>6.1%} -> {best:>6.1%}   {100*(best-alone):>+6.1f}")

# ── IV-C ────────────────────────────────────────────────────────────────
print("\n" + "=" * 74)
print("IV-C  COMPOSITION ORDER")
print("=" * 74)
c = p[p.config.str.count(r"\|") == 2].copy()
c["order"] = c.config.str.split("|").str[2]
o = c.groupby(["model", "order"]).success.mean().unstack() * 100
o["delta"] = o["libra_then_utos"] - o["utos_then_libra"]
print(o.round(1).sort_values("delta", ascending=False).to_string())

# ── IV-D ────────────────────────────────────────────────────────────────
print("\n" + "=" * 74)
print("IV-D  EVALUATOR")
print("=" * 74)
JUDGE_CACHE = "results/judge_cache.parquet"
try:
    x = pd.read_parquet(JUDGE_CACHE)
except FileNotFoundError:
    print(f"{JUDGE_CACHE} not found -- rebuild it from the stage CSVs")
    x = pd.DataFrame()

if not x.empty:
    lg = x.jailbreak_success_llama_guard.fillna(False).astype(bool)
    hb = x.jailbreak_success_harmbench.fillna(False).astype(bool)
    dis = lg != hb
    print(f"rows scored              : {len(x):,}")
    print(f"classifier disagreement  : {dis.mean():.1%}")
    print(f"  LG harmful, HB safe    : {(lg & ~hb).sum():,}")
    print(f"  HB harmful, LG safe    : {(hb & ~lg).sum():,}  "
          f"({(hb & ~lg).sum()/max(dis.sum(),1):.1%} of disagreements)")
    if "harmbench_confidence" in x.columns:
        print(f"  HB confidence on disagreements, median: "
              f"{x.loc[dis, 'harmbench_confidence'].median():.4f}")
    if "hit_token_cap" in x.columns:
        cap = x.hit_token_cap.fillna(False).astype(bool)
        print(f"  truncated | disagree     : {cap[dis].mean():.1%}")
        print(f"  truncated | agree        : {cap[~dis].mean():.1%}")
    print(f"\npooled ASR, strict  (LG AND HB) : {(lg & hb).mean():.1%}")
    print(f"pooled ASR, lenient (LG OR  HB) : {(lg | hb).mean():.1%}")
    if "jailbreak_success_ensemble" in x.columns:
        ens = x.jailbreak_success_ensemble.fillna(False).astype(bool)
        print(f"pooled ASR, 2-of-3 ensemble     : {ens.mean():.1%}")
        print(f"  ensemble successes decided by the heuristic: "
              f"{(ens & ~(lg & hb)).sum()/max(ens.sum(),1):.1%}")

# ── floor cases ─────────────────────────────────────────────────────────
print("\n" + "=" * 74)
print("FLOOR CASES")
print("=" * 74)
for m in FLOOR:
    g = d[d.model == m]
    if g.empty:
        continue
    a = g.groupby("config").success.mean()
    n = g.instruction_id.nunique()
    print(f"{m:<14} best {a.max():>5.1%}  ({int(round(a.max()*n))}/{n} instructions)"
          f"   configs > 0: {(a > 0).sum()}/103")