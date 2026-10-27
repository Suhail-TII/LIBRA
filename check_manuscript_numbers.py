#!/usr/bin/env python3
"""
check_manuscript_numbers.py -- read the numbers out of the manuscript source
and check them against the released result files.

This is the check make_tables.py does not perform. make_tables.py compares
CSV-derived values against constants embedded in itself: it will pass even if
someone edits a figure in the .tex, because it never opens the .tex. This
script closes that gap. It parses the LaTeX, pulls out the entries it can
locate unambiguously, recomputes each one from results/, and reports.

What it checks, by locating the table environments by \\label and the prose
figures by their surrounding wording:

  Table  (tab:regret_perfold)   per-target regret at 960, every cell
  Table  (tab:ksweep)           the screening-size sweep, every cell
  prose  Figure 3 / Section V-D LIBRA, halving, singles, UCB1, random
  prose  Section VI-A           normalised regret per target
  caption Figure 3              the two headline regrets

It is deliberately conservative. A number it cannot locate unambiguously is
reported as NOT FOUND rather than silently skipped, so the summary line
distinguishes "checked and agrees" from "never looked at".

    python check_manuscript_numbers.py --tex Chapters --results results
    python check_manuscript_numbers.py --tex Chapters --results results -v
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd

METHODS = ["LIBRA", "full_space_halving", "singles_sweep", "ucb", "random"]
TARGETS = [  # (internal, label as it appears in the tables)
    ("gpt-5.1", "GPT-5.1"),
    ("Llama-2-7B-Chat", "Llama-2-7B"),
    ("Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
    ("Qwen3-14B", "Qwen3-14B"),
    ("Ministral-3-8B-Instruct", "Ministral-3-8B"),
]
T6_DESIGNS = ["11_coverage", "22_coverage", "22_main", "33_coverage", "44_coverage"]

results: list[tuple[str, str, float | None, float | None]] = []


def record(what: str, status: str, found=None, expect=None) -> None:
    results.append((what, status, found, expect))


def num(s: str) -> float | None:
    s = s.replace("\\%", "").replace("$", "").replace("\\textbf{", "").replace("}", "")
    s = s.replace("\\,", "").replace("{,}", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def table_body(tex: str, label: str) -> list[list[str]] | None:
    """Rows of the tabular carrying this \\label, between \\midrule and \\bottomrule."""
    i = tex.find("\\label{" + label + "}")
    if i < 0:
        return None
    seg = tex[i:]
    end = seg.find("\\end{table}")
    seg = seg[:end if end > 0 else len(seg)]
    m = re.search(r"\\midrule(.*?)\\bottomrule", seg, re.S)
    if not m:
        return None
    rows = []
    for raw in re.split(r"\\\\", m.group(1)):
        raw = re.sub(r"(?<!\\)%.*", "", raw)   # not \%, which is a literal percent
        if "&" not in raw:
            continue
        rows.append([c.strip() for c in raw.split("&")])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default="Chapters", help="directory of .tex chapters")
    ap.add_argument("--results", default="results")
    ap.add_argument("--tol", type=float, default=0.006,
                    help="tolerance in points; default catches a wrong 2nd decimal")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    texdir, res = Path(a.tex), Path(a.results)
    tex = "\n".join(p.read_text(errors="replace") for p in sorted(texdir.glob("*.tex")))
    if not tex.strip():
        sys.exit(f"no .tex found under {texdir}")

    ps = pd.read_csv(res / "per_seed_regrets.csv")
    ps["pts"] = ps.regret.astype(float) * 100
    blocks = ps.groupby(["method", "budget", "seed"], as_index=False).pts.mean()
    fig = blocks.groupby(["method", "budget"]).pts.mean()
    t5 = ps[ps.budget == 960].pivot_table(index="model", columns="method", values="pts")
    agg = pd.read_csv(res / "reviewer_analyses.csv")
    star = agg.drop_duplicates("model").set_index("model").c_star_asr
    t6 = pd.read_csv(res / "table6_per_seed.csv").pivot_table(
        index="budget", columns="design", values="regret_points")

    # ---- Table: per-target regret at 960 -------------------------------
    rows = table_body(tex, "tab:regret_perfold")
    if rows is None:
        record("Table tab:regret_perfold", "NOT FOUND")
    else:
        bylabel = {r[0].strip(): r for r in rows}
        for internal, label in TARGETS:
            r = bylabel.get(label)
            if r is None or len(r) < 7:
                record(f"Table 5 / {label}", "NOT FOUND"); continue
            for k, cell in zip(METHODS, r[2:7]):
                got, want = num(cell), float(t5.loc[internal, k])
                if got is None:
                    record(f"Table 5 / {label} / {k}", "UNPARSED")
                else:
                    record(f"Table 5 / {label} / {k}",
                           "ok" if abs(got - want) <= a.tol else "MISMATCH", got, want)
            cs = num(r[1])
            if cs is not None:
                want = float(star[internal]) * 100
                record(f"Table 5 / {label} / c*",
                       "ok" if abs(cs - want) <= 0.06 else "MISMATCH", cs, want)
            else:
                record(f"Table 5 / {label} / c*", "UNPARSED")

    # ---- Table: screening-size sweep ------------------------------------
    rows = table_body(tex, "tab:ksweep")
    if rows is None:
        record("Table tab:ksweep", "NOT FOUND")
    else:
        bybudget = {}
        for r in rows:
            b = num(r[0])
            if b is None or int(b) not in t6.index:
                record("Table 6 / unexpected or unparsed budget", "UNPARSED")
                continue
            b = int(b)
            if b in bybudget:
                record(f"Table 6 / duplicate budget {b}", "UNPARSED")
            bybudget[b] = r
        for b in sorted(t6.index):
            r = bybudget.get(b)
            if r is None or len(r) != 6:
                record(f"Table 6 / budget {b}", "NOT FOUND")
                continue
            for d, cell in zip(T6_DESIGNS, r[1:6]):
                got = num(re.sub(r"\$\^\{?\\?\*?\}?\$|\^\{\*\}|\\ast", "", cell))
                if got is None:
                    record(f"Table 6 / {int(b)} / {d}", "UNPARSED"); continue
                want = float(t6.loc[int(b), d])
                record(f"Table 6 / {int(b)} / {d}",
                       "ok" if abs(got - want) <= a.tol else "MISMATCH", got, want)

    # ---- prose: Figure 3 headline regrets --------------------------------
    for pat, meth, bud in [
        (r"reaches \$6\.(\d)\$ points at 240 queries and \$2\.(\d)\$ at 960", None, None),
    ]:
        m = re.search(pat, tex)
        if m:
            for digit, b in zip(m.groups(), (240, 960)):
                got = float(f"{6 if b == 240 else 2}.{digit}")
                want = float(fig.loc[("LIBRA", b)])
                record(f"Fig 3 caption / LIBRA at {b}",
                       "ok" if abs(got - want) <= 0.05 else "MISMATCH", got, want)
        else:
            record("Fig 3 caption / LIBRA headline", "NOT FOUND")

    # ---- prose: Section V-D figures --------------------------------------
    prose = [
        (r"within \$6\.(\d)\$ points of the optimum", "LIBRA", 240, 6),
        (r"19\\times\$ fewer, within \$2\.(\d)\$ points", "LIBRA", 960, 2),
        (r"singles sweep has lower aggregate mean regret: \$12\.(\d)\$ points"
         r" against LIBRA's \$1(\d)\.(\d)\$", None, None, None),
        (r"costs \$40\.(\d)\$ points at 120 queries", "full_space_halving", 120, 40),
        (r"and \$7\.(\d)\$ at 240", "full_space_halving", 240, 7),
        (r"closes to \$2\.(\d)\$ at 960", "full_space_halving", 960, 2),
        (r"improves steadily to \$4\.(\d)\$ points at 960", "ucb", 960, 4),
        (r"but costs \$34\.(\d)\$ at 120", "ucb", 120, 34),
    ]
    for pat, meth, b, whole in prose:
        m = re.search(pat, tex)
        if not m:
            record(f"prose / {pat[:34]}", "NOT FOUND"); continue
        if meth is None:   # the singles-vs-ours sentence carries two numbers
            g = m.groups()
            for val, mm, bb in ((float(f"12.{g[0]}"), "singles_sweep", 120),
                                (float(f"{g[1]}{g[2] and ''}") if False else float(f"1{g[1]}.{g[2]}"),
                                 "LIBRA", 120)):
                want = float(fig.loc[(mm, bb)])
                record(f"prose / {mm} at {bb}",
                       "ok" if abs(val - want) <= 0.05 else "MISMATCH", val, want)
            continue
        got = float(f"{whole}.{m.group(1)}")
        want = float(fig.loc[(meth, b)])
        record(f"prose / {meth} at {b}",
               "ok" if abs(got - want) <= 0.05 else "MISMATCH", got, want)

    # ---- prose: the Ministral out-of-selection claim in Section III ------
    m = re.search(r"recovers one within \$(\d)\.(\d)\$ points of it after 960 queries", tex)
    if m:
        got = float(f"{m.group(1)}.{m.group(2)}")
        want = float(t5.loc["Ministral-3-8B-Instruct", "LIBRA"])
        record("III-A / Ministral at 960",
               "ok" if abs(got - want) <= 0.05 else "MISMATCH", got, want)
    else:
        record("III-A / Ministral at 960", "NOT FOUND")

    # ---- prose: normalised regret ---------------------------------------
    m = re.search(r"forfeits \$10\.(\d)\\%\$ and \$10\.(\d)\\%\$", tex)
    if m:
        for digit, internal, label in ((m.group(1), "Llama-2-7B-Chat", "Llama-2-7B"),
                                       (m.group(2), "gpt-5.1", "GPT-5.1")):
            got = float(f"10.{digit}")
            want = float(t5.loc[internal, "LIBRA"]) / (float(star[internal]) * 100) * 100
            record(f"VI-A normalised / {label}",
                   "ok" if abs(got - want) <= 0.06 else "MISMATCH", got, want)
    else:
        record("VI-A normalised regret pair", "NOT FOUND")

    # The frozen manuscript check covers 79 entries. Missing coverage fails.
    if len(results) != 79:
        record(f"Expected 79 checks; located {len(results)}", "NOT FOUND")

    # ---- report ----------------------------------------------------------
    ok = sum(1 for _, s, *_ in results if s == "ok")
    bad = [r for r in results if r[1] == "MISMATCH"]
    miss = [r for r in results if r[1] in ("NOT FOUND", "UNPARSED")]

    print("=" * 72)
    print("MANUSCRIPT NUMBERS vs RELEASED RESULTS")
    print("=" * 72)
    if a.verbose:
        for what, status, got, want in results:
            g = "" if got is None else f"{got:8.2f}"
            w = "" if want is None else f"{want:8.2f}"
            print(f"  {status:<9} {what:<42}{g}{w}")
    print(f"\n  agree          {ok}")
    print(f"  MISMATCH       {len(bad)}")
    print(f"  not located    {len(miss)}")
    for what, _, got, want in bad:
        print(f"    MISMATCH  {what:<42} manuscript {got}  results {want:.4f}")
    for what, status, *_ in miss:
        print(f"    {status:<10}{what}")
    if not bad:
        print("\n  every located manuscript number matches the released results.")
    print("  Numbers reported as 'not located' were NOT checked: this script")
    print("  reads the .tex, so a reworded sentence hides its number from it.")
    return 1 if bad or miss else 0


if __name__ == "__main__":
    sys.exit(main())
