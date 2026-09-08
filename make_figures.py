#!/usr/bin/env python3
"""
make_figures.py — the two figures for the paper.

  fig_regret.pdf   regret against query budget, all methods. Single column.
                   Replaces the budget table: a table of six rows by five
                   columns says less than the shape of the curves, and the
                   crossover with the singles sweep at low budget is visible
                   at a glance rather than needing a sentence.

  fig_heatmap.pdf  the 103 x 5 grid. Every configuration on every target,
                   sorted so structure is visible. This is the measurement the
                   whole paper rests on and it has never been shown directly.

Both read only the parquet caches, so they run anywhere in seconds.

    python make_figures.py --outdir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter, NullLocator
from matplotlib.patches import Patch

POOL = ["Llama-2-7B-Chat", "Llama-3.1-8B-Instruct", "Qwen3-14B",
        "gpt-5.1", "Ministral-3-8B-Instruct"]
SHORT = {"Llama-2-7B-Chat": "Llama-2-7B",
         "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
         "Qwen3-14B": "Qwen3-14B",
         "gpt-5.1": "GPT-5.1",
         "Ministral-3-8B-Instruct": "Ministral-3-8B"}

# IEEE two-column: 3.5in single, 7.16in double. Keep fonts at 8pt so they are
# not smaller than the caption after any scaling.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.5,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — regret vs budget
# ═══════════════════════════════════════════════════════════════════════════

def fig_regret(results_csv: str, outdir: Path, seeds_csv: str | None = None):
    """
    One panel, log-x, with shaded interquartile ranges where per-seed values
    are available.
 
    The band is the 25th-75th percentile over seeds, pooled across the five
    targets. It is deliberately not a confidence interval on the mean: the
    quantity a reader cares about is how much an individual run varies, since
    an adversary gets one run, not two hundred.
    """
    r = pd.read_csv(results_csv)
    piv = r.pivot_table(index="budget", columns="method", values="regret") * 100
 
    bands = None
    if seeds_csv and Path(seeds_csv).exists():
        s = pd.read_csv(seeds_csv)
        # Average across targets WITHIN each seed first. Pooling (target, seed)
        # pairs would show between-target spread, which is large because the
        # targets have ceilings from 23.6% to 90.6%, and says nothing about
        # whether the difference between methods is real.
        per_seed = (s.groupby(["method", "budget", "seed"]).regret
                     .mean().reset_index())
        g = per_seed.groupby(["method", "budget"]).regret
        m, sd, n = g.mean() * 100, g.std() * 100, g.count()
        half = 1.96 * sd / np.sqrt(n)
        bands = pd.DataFrame({0.25: m - half, 0.75: m + half})
 
    style = {
        "LIBRA":         dict(color="#1f5c3d", marker="o", lw=2.0, zorder=5),
        "singles_sweep": dict(color="#8c2f1f", marker="s", ls="--"),
        "ucb":           dict(color="#31527a", marker="^", ls="-."),
        "random":        dict(color="#7a7a7a", marker="v", ls=":"),
        "race_only":     dict(color="#c08a5a", marker="d", ls=(0, (3, 1, 1, 1))),
    }
    label = {"LIBRA": "LIBRA", "singles_sweep": "Singles sweep",
             "ucb": "UCB1", "random": "Random search",
             "race_only": "Race, no screening"}
 
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
 
    order = piv.iloc[-1].sort_values().index
    for m in order:
        if m not in style:
            continue
        # band first so the line sits on top of it
        if bands is not None and m in bands.index.get_level_values(0):
            b = bands.loc[m]
            ax.fill_between(b.index, b[0.25], b[0.75],
                            color=style[m]["color"], alpha=0.13, lw=0, zorder=2)
        ax.plot(piv.index, piv[m], label=label[m], **style[m])
 
    ax.axhline(32.6, color="#b0b0b0", lw=0.9, zorder=1)
    ax.text(piv.index[0], 31.6, "Transfer (0 queries)", ha="left", va="top",
            fontsize=6.5, color="#707070")
 
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(list(piv.index))
    ax.set_xticklabels([str(b) for b in piv.index])
    ax.set_xlabel("Query budget")
    ax.set_ylabel("Mean regret (points of ASR)")
    ax.set_ylim(0, 37)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, ncol=3, handlelength=1.8, columnspacing=1.2,
              loc="upper center", bbox_to_anchor=(0.5, -0.18))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
 
    out = outdir / "fig_regret.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  {out}" + ("  (with IQR bands)" if bands is not None else
                        "  (no per-seed file -- lines only)"))


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — the design space
# ═══════════════════════════════════════════════════════════════════════════

def fig_heatmap(cache: str, outdir: Path):
    """
    103 configurations x 5 targets.

    Configurations are ordered by their mean ASR across targets, which makes
    two things visible that the tables state but do not show: the columns are
    not rank-correlated (a configuration high on one target is often low on
    another), and each target's optimum sits at a different row.
    """
    d = pd.read_parquet(cache)
    d = d[d.model.isin(POOL)]
    a = d.groupby(["config", "model"]).success.mean().unstack() * 100
    a = a[POOL]

    a["_order"] = a.mean(axis=1)
    a = a.sort_values("_order", ascending=False).drop(columns="_order")

    kind = ["single" if "|" not in c else "composed" for c in a.index]

    cmap = LinearSegmentedColormap.from_list(
        "libra", ["#ffffff", "#dce6f0", "#7a9cc6", "#31527a", "#1a2c4a"])

    fig, (axl, ax) = plt.subplots(
        1, 2, figsize=(3.5, 4.4), gridspec_kw={"width_ratios": [0.06, 1]})

    # left strip: single component vs composed cell
    axl.imshow(np.array([[0 if k == "single" else 1] for k in kind]),
               aspect="auto", cmap=LinearSegmentedColormap.from_list(
                   "k", ["#c08a5a", "#e8eef6"]), vmin=0, vmax=1)
    axl.set_xticks([]); axl.set_yticks([])
    for s in axl.spines.values():
        s.set_linewidth(0.4)
    
    axl.legend(handles=[Patch(facecolor="#c08a5a", label="single"),
                        Patch(facecolor="#e8eef6", ec="#b0b0b0", lw=0.4,
                              label="composed")],
               loc="lower right", bbox_to_anchor=(0.5, -0.05),
               frameon=False, fontsize=6, handlelength=0.9,
               handleheight=0.9, labelspacing=0.3, borderpad=0)
    axl.set_ylabel("103 configurations, ordered by mean ASR", fontsize=7.5)

    im = ax.imshow(a.values, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(len(POOL)))
    ax.set_xticklabels([SHORT[m] for m in POOL], rotation=35,
                       ha="right", fontsize=7)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)

    # mark each target's optimum
    for j, m in enumerate(POOL):
        i = int(np.argmax(a[m].values))
        ax.plot(j, i, marker="*", ms=7, color="#8c2f1f",
                mec="white", mew=0.5, zorder=5)

    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Attack success rate (%)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5, width=0.5)
    cb.outline.set_linewidth(0.4)

    fig.subplots_adjust(wspace=0.04)
    out = outdir / "fig_heatmap.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"  {out}")

    # numbers worth quoting in the caption
    print("\n  for the caption:")
    for m in POOL:
        i = int(np.argmax(a[m].values))
        print(f"    {SHORT[m]:<15} c* at row {i+1}/103, "
              f"{a[m].iloc[i]:.1f}%  ({'single' if kind[i]=='single' else 'composed'})")
    corr = a.corr(method="spearman").values
    off = corr[np.triu_indices_from(corr, k=1)]
    print(f"    pairwise column rank correlation: "
          f"{off.min():.2f} to {off.max():.2f}, mean {off.mean():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/replay_cache.parquet")
    ap.add_argument("--results", default="results/reviewer_analyses.csv")
    ap.add_argument("--seeds-csv", default="results/regret_seeds.csv")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    print("writing:")
    fig_regret(args.results, out, args.seeds_csv)
    fig_heatmap(args.cache, out)


if __name__ == "__main__":
    main()
