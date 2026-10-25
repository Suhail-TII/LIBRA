#!/usr/bin/env python3
"""
make_tables.py -- regenerate Table 5 and Table 6 from the result CSVs, and
verify every number the manuscript quotes against them.

The figures are not built here. make_figures.py owns fig_regret.pdf and
fig_heatmap.pdf and is left alone; this script only writes LaTeX tables and
checks arithmetic, so the two can be run in either order.

Nothing here runs an experiment. It reads only what the sweeps already
wrote, so it is safe to re-run and cannot change a result:

    results/per_seed_regrets.csv    <- reviewer_analyses.py
    results/reviewer_analyses.csv   <- reviewer_analyses.py  (c* per target)
    results/table6_per_seed.csv     <- table6_paired.py
    results/table6_paired.csv       <- table6_paired.py      (Holm stars)
    results/table6_designs.json     <- table6_paired.py      (pair coverage)

Outputs tables/table5.tex and tables/table6.tex, then prints a VERIFY block
comparing the regenerated values against the numbers written into the .tex.
Exit status is 1 if any disagree. A FAIL is a number to fix in the
manuscript, not a bug to work around: the EXPECT_* tables below are there to
be read, never edited into silence.

    python make_tables.py
    python make_tables.py --no-verify
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- constants

# Fixed hue order, assigned to the entity and never cycled. Validated
# colourblind-safe against a white surface; every series additionally carries
# its own marker and dash pattern, so identity never rests on colour alone
# and the figure survives greyscale printing.
SERIES = [          # key, label: fixes the column order of Table 5
    ("LIBRA",              "LIBRA"),
    ("full_space_halving", "Full-space halving"),
    ("singles_sweep",      "Singles sweep"),
    ("ucb",                "UCB1"),
    ("random",             "Random search"),
]

TARGETS = [  # internal name -> label, in the order Table 5 prints them
    ("gpt-5.1",                 "GPT-5.1"),
    ("Llama-2-7B-Chat",         "Llama-2-7B"),
    ("Llama-3.1-8B-Instruct",   "Llama-3.1-8B"),
    ("Qwen3-14B",               "Qwen3-14B"),
    ("Ministral-3-8B-Instruct", "Ministral-3-8B"),
]

T6_COLS = [  # design key -> (header, framing-order pairs shown in the stub)
    ("11_coverage", r"$k_s{=}11$"),
    ("22_coverage", r"$k_s{=}22$"),
    ("22_main",     r"$\mathcal{S}^{\dagger}$"),
    ("33_coverage", r"$k_s{=}33$"),
    ("44_coverage", r"$k_s{=}44$"),
]

# What the manuscript says. Edit these only to match a value you have decided
# to change in the .tex, never to silence a FAIL.
EXPECT_FIG = {  # method -> {budget: points}
    "LIBRA":              {120: 17.06, 240: 6.43, 360: 5.49, 480: 3.37, 720: 2.73, 960: 2.29},
    "full_space_halving": {120: 40.36, 240: 7.95, 360: 5.11, 480: 3.70, 720: 3.00, 960: 2.57},
    "singles_sweep":      {120: 12.00, 240: 10.26, 360: 9.59, 480: 8.94, 720: 8.62, 960: 8.31},
    "ucb":                {120: 34.20, 240: 19.30, 360: 12.31, 480: 8.81, 720: 5.85, 960: 4.38},
    "random":             {120: 32.80, 240: 25.98, 360: 21.19, 480: 17.46, 720: 12.93, 960: 9.27},
}
EXPECT_T5 = {  # target -> (LIBRA, halving, singles, ucb, random) at 960
    "gpt-5.1":                 (2.42,  1.59, 19.29, 3.24,  4.35),
    "Llama-2-7B-Chat":         (3.11, 10.17, 18.64, 10.06, 10.86),
    "Llama-3.1-8B-Instruct":   (1.92,  0.23,  0.00, 3.10, 11.78),
    "Qwen3-14B":               (1.54,  0.10,  0.54, 1.86, 10.44),
    "Ministral-3-8B-Instruct": (2.44,  0.78,  3.08, 3.65,  8.90),
}
EXPECT_NORM = {  # target -> normalised regret %, Section VI-A
    "Llama-2-7B-Chat": 10.4, "Llama-3.1-8B-Instruct": 2.1, "Qwen3-14B": 1.7,
    "gpt-5.1": 10.3, "Ministral-3-8B-Instruct": 2.7,
}
EXPECT_T6 = {  # budget -> (11, 22_cov, S dagger, 33, 44)
    120:  (14.44, 14.85, 17.06, 24.30, 25.82),
    240:  ( 7.01,  6.14,  6.43,  7.03,  8.17),
    360:  ( 5.14,  4.87,  5.49,  5.50,  5.41),
    480:  ( 3.14,  3.17,  3.37,  3.50,  3.00),
    720:  ( 2.53,  2.72,  2.73,  2.89,  2.52),
    960:  ( 1.89,  2.39,  2.29,  2.55,  1.95),
    1440: ( 0.99,  1.21,  1.16,  1.30,  1.19),
}
EXPECT_MINISTRAL = {  # Section V-F: 11_coverage vs 22_main on the one target
    240: (6.33, 8.93), 360: (3.84, 7.75),
}

FAILURES: list[str] = []


def check(label: str, got: float, want: float, tol: float = 0.005) -> None:
    ok = got is not None and abs(got - want) <= tol
    if not ok:
        FAILURES.append(label)
    g = "   n/a" if got is None else f"{got:6.2f}"
    print(f"    {'PASS' if ok else 'FAIL'}  {label:<44} got {g}  want {want:6.2f}")


# ------------------------------------------------------------------ loading

def blocks(ps: pd.DataFrame) -> pd.DataFrame:
    """
    One block per (method, budget, seed): the mean over targets under that
    seed. Blocking this way removes between-target variance, so the interval
    describes seed-to-seed spread of the procedure, which is what the figure
    caption claims. It is the same blocking the paired tests use.
    """
    return (ps.groupby(["method", "budget", "seed"], as_index=False)
              .regret_points.mean())


def load(resdir: Path) -> dict:
    d = {}
    f = resdir / "per_seed_regrets.csv"
    if not f.exists():
        sys.exit(f"missing {f} -- run reviewer_analyses.py first")
    ps = pd.read_csv(f)
    # the sweep stores regret as a fraction of ASR; the paper reports points
    ps["regret_points"] = ps.regret.astype(float) * 100.0
    d["per_seed"] = ps
    d["blocks"] = blocks(ps)
    print(f"  per_seed_regrets.csv   {len(ps):>8,} rows  "
          f"{ps.seed.nunique()} seeds  "
          f"budgets {sorted(int(b) for b in ps.budget.unique())}")

    f = resdir / "reviewer_analyses.csv"
    d["agg"] = pd.read_csv(f) if f.exists() else None
    if d["agg"] is not None:
        print(f"  reviewer_analyses.csv  {len(d['agg']):>8,} rows")

    for name, key in [("table6_per_seed.csv", "t6_ps"),
                      ("table6_paired.csv", "t6_p")]:
        f = resdir / name
        d[key] = pd.read_csv(f) if f.exists() else None
        if d[key] is not None:
            print(f"  {name:<22} {len(d[key]):>8,} rows")
    f = resdir / "table6_designs.json"
    d["t6_d"] = json.loads(f.read_text()) if f.exists() else None
    return d


# ------------------------------------------------------------------ table 5

def table5(d: dict, budget: int, out: Path) -> pd.DataFrame:
    ps = d["per_seed"]
    m = (ps[ps.budget == budget]
         .pivot_table(index="model", columns="method", values="regret_points"))

    star = {}
    if d["agg"] is not None and "c_star_asr" in d["agg"].columns:
        star = (d["agg"].drop_duplicates("model")
                .set_index("model").c_star_asr.to_dict())

    keys = [k for k, _ in SERIES]
    lines = [
        r"\begin{table}[!t]", r"\centering",
        r"\caption{Per-target regret at " + str(budget) + r" queries. Every "
        r"baseline forfeits more than ten points on at least one target. The "
        r"procedure's regret spans $%.2f$ to $%.2f$ points and is the only "
        r"method that stays within four points everywhere.}",
        r"\label{tab:regret_perfold}", r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l r r r r r r@{}}", r"\toprule",
        r"Target & $c^{\star}$ & LIBRA & \makecell{Full-space\\halving} & "
        r"\makecell{Singles\\sweep} & UCB1 & \makecell{Random\\search} \\",
        r"\midrule",
    ]
    lo = hi = None
    for internal, label in TARGETS:
        if internal not in m.index:
            print(f"    note: '{internal}' absent from the sweep, row skipped")
            continue
        vals = [float(m.loc[internal, k]) if k in m.columns else np.nan
                for k in keys]
        best = int(np.nanargmin(vals))
        lo = vals[0] if lo is None else min(lo, vals[0])
        hi = vals[0] if hi is None else max(hi, vals[0])
        cs = (f"{star[internal] * 100:.1f}\\%" if internal in star else "--")
        # regret is non-negative by construction; kill the "-0.00" that float
        # noise produces on a cell whose true value is exactly zero
        vals = [0.0 if abs(v) < 0.005 else v for v in vals]
        cells = [(r"\textbf{%.2f}" % v) if i == best else (f"{v:.2f}")
                 for i, v in enumerate(vals)]
        lines.append(f"{label:<15} & {cs} & " + " & ".join(cells) + r" \\")
    lines[2] = lines[2] % (lo, hi)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"    wrote {out}   (LIBRA span {lo:.2f}-{hi:.2f})")
    return m


# ------------------------------------------------------------------ table 6

def table6(d: dict, out: Path) -> pd.DataFrame | None:
    if d["t6_ps"] is None:
        print("    table6_per_seed.csv absent, Table 6 skipped")
        return None
    ps = d["t6_ps"]
    means = ps.pivot_table(index="budget", columns="design",
                           values="regret_points")

    star: set[tuple[int, str]] = set()
    if d["t6_p"] is not None and "holm_p" in d["t6_p"].columns:
        for _, r in d["t6_p"].iterrows():
            if r.holm_p < 0.05:
                star.add((int(r.budget), str(r.comparison).split(" minus ")[0]))
    n_comp = 0 if d["t6_p"] is None else len(d["t6_p"])

    fo = {}
    if d["t6_d"]:
        fo = {k: v["framing_order_pairs"] for k, v in d["t6_d"].items()}

    hdr = " & ".join(h for _, h in T6_COLS)
    stub = " & ".join(f"({fo.get(k, '?')}/22)" for k, _ in T6_COLS)
    lines = [
        r"\begin{table}[!t]", r"\centering",
        r"\caption{Mean regret in points across the five targets, sweeping the",
        r"screening size against budget. $\mathcal{S}^{\dagger}$ is the design the",
        r"procedure uses; the others are constructed at each size to maximise pair",
        r"coverage. Parenthesised values are distinct (framing, order) pairs out of 22.",
        r"$^{*}$ marks a paired difference from $\mathcal{S}^{\dagger}$ over "
        + str(int(ps.seed.nunique())) + r" seed",
        r"blocks that survives Holm adjustment across all " + str(n_comp)
        + r" comparisons. Cell lists",
        r"and their hashes are released with the code.}",
        r"\label{tab:ksweep}", r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}r r r r r r@{}}", r"\toprule",
        r" & " + hdr + r" \\",
        r"Budget & " + stub + r" \\",
        r"\midrule",
    ]
    for b in sorted(means.index):
        cells = []
        for k, _ in T6_COLS:
            if k not in means.columns:
                cells.append("--"); continue
            v = f"{means.loc[b, k]:5.2f}"
            cells.append(v + (r"$^{*}$" if (b, k) in star else ""))
        lines.append(f"{int(b):>4} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"    wrote {out}   ({len(star)} starred cells, "
          f"Holm across {n_comp})")
    return means


# ------------------------------------------------------------------- verify

def verify(d: dict, fig_means: pd.DataFrame, t5: pd.DataFrame,
           t6: pd.DataFrame | None) -> None:
    print("\n" + "=" * 74)
    print("VERIFY  regenerated values against the numbers in the manuscript")
    print("=" * 74)

    print("\n  Figure 3 and Section V-D")
    for key, label in SERIES:
        for b, want in EXPECT_FIG[key].items():
            got = (float(fig_means.loc[(key, b)])
                   if (key, b) in fig_means.index else None)
            check(f"{label} at {b}", got, want)

    print("\n  Table 5, per target at 960")
    keys = [k for k, _ in SERIES]
    for internal, label in TARGETS:
        for k, want in zip(keys, EXPECT_T5[internal]):
            got = (float(t5.loc[internal, k])
                   if internal in t5.index and k in t5.columns else None)
            check(f"{label} / {k}", got, want)

    print("\n  Section VI-A, normalised regret at 960")
    if d["agg"] is not None and "c_star_asr" in d["agg"].columns:
        star = d["agg"].drop_duplicates("model").set_index("model").c_star_asr
        for internal, label in TARGETS:
            got = None
            if internal in t5.index and internal in star.index:
                got = float(t5.loc[internal, "LIBRA"]) / (star[internal] * 100) * 100
            check(f"{label} normalised %", got, EXPECT_NORM[internal], tol=0.06)
    else:
        print("    skipped: reviewer_analyses.csv has no c_star_asr column")

    if t6 is not None:
        print("\n  Table 6, screening-size sweep")
        for b, wants in EXPECT_T6.items():
            for (k, _), want in zip(T6_COLS, wants):
                got = (float(t6.loc[b, k])
                       if b in t6.index and k in t6.columns else None)
                check(f"budget {b} / {k}", got, want)

        print("\n  Section V-F, Ministral screening size")
        ps = d["t6_ps"]
        mini = ps[ps.model == "Ministral-3-8B-Instruct"]
        for b, (w11, w22) in EXPECT_MINISTRAL.items():
            for design, want in (("11_coverage", w11), ("22_main", w22)):
                s = mini[(mini.budget == b) & (mini.design == design)]
                got = float(s.regret_points.mean()) if len(s) else None
                check(f"Ministral {design} at {b}", got, want, tol=0.01)

    print("\n" + "-" * 74)
    if FAILURES:
        print(f"  {len(FAILURES)} MISMATCH(ES) -- the .tex and the data disagree:")
        for f in FAILURES:
            print(f"    - {f}")
        print("\n  Fix the manuscript to the regenerated value, or say why the")
        print("  data moved. Do not edit EXPECT_* to make this quiet.")
    else:
        print("  every manuscript number reproduces from the released CSVs")
    print("-" * 74)


# --------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--tabdir", default="tables")
    ap.add_argument("--table-budget", type=int, default=960)
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    res, tabdir = Path(a.results), Path(a.tabdir)

    print("LOADING")
    d = load(res)

    print("\nTABLE 5")
    t5 = table5(d, a.table_budget, tabdir / "table5.tex")

    print("\nTABLE 6")
    t6 = table6(d, tabdir / "table6.tex")

    fig_means = (d["blocks"].groupby(["method", "budget"]).regret_points.mean())

    if not a.no_verify:
        verify(d, fig_means, t5, t6)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())