#!/usr/bin/env python3
"""
analyse_search_space.py

Two jobs.

1. Build the SEARCH SPACE the selection algorithm actually searches, and the
   ground truth c* it is measured against.

   The Stage B grid is 88 composed cells. Defining c* over those alone is
   wrong: on Llama-3.1 the best composed cell reaches 75.6% while
   plinius_format_hijack ALONE reaches 90.0%, so a regret computed against
   the 88-cell optimum would be measured against a target that is not the
   best available configuration. The adversary does not know in advance
   whether to compose, so "do not compose" must be in the space.

       88 composed cells (11 framings x 4 templates x 2 orders)
     + 11 framing-only
     +  4 template-only
     = 103 configurations

   All three come from the same 180 probe instructions, the same judges and
   the same 512-token cap, so they are directly comparable and nothing needs
   re-running.

2. Characterise WHEN composition helps.

   Llama-2  : best component 11.7% -> best cell 30.0%   (+18.3)
   Llama-3.1: best component 90.0% -> best cell 75.6%   (-14.4, 0/88 beat it)

   The hypothesis is that composition helps only where no single component
   already works, and interferes once one does. This script tests it across
   whatever models are complete, and decomposes the interference: for each
   framing, its standalone ASR against its own composed cells.

Usage
-----
    python analyse_search_space.py
    python analyse_search_space.py --metric jailbreak_success_ensemble
    python analyse_search_space.py --out results/search_space.csv
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def _load(patterns, key, metric):
    """Load and dedupe one stage's rows, returning only usable ones."""
    frames = []
    for pat in patterns:
        for f in glob.glob(pat):
            if "profile" in f or "runmeta" in f:
                continue
            d = pd.read_csv(f)
            if key not in d.columns or metric not in d.columns:
                continue
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d = d[~d.get("api_error", pd.Series(False, index=d.index)).fillna(False)]
    return d.drop_duplicates(subset=["model", key, "instruction_id"], keep="last")

def _apply_decoded(ce: pd.DataFrame, metric: str,
                   decoded_glob: str = "results/stage_b_decoded/decoded_*.csv"
                   ) -> pd.DataFrame:
    """
    Replace raw judge labels with decoded ones for encoded responses.

    Two framings force leetspeak output: plinius_l33tspeak and
    plinius_full_openai. Both classifiers misread that encoding, and NOT
    symmetrically -- the GPT models REFUSE in leetspeak (Llama Guard flagged
    78-85% of encoded GPT responses raw, 9-33% decoded) while the open-weight
    models genuinely COMPLY in leetspeak and are read correctly. Since
    plinius_l33tspeak cells were the recorded c* for all three GPT models, the
    ground truth is wrong for half the pool until this is applied.

    Only rows with is_encoded == True are overridden.
    """
    files = sorted(glob.glob(decoded_glob))
    if not files:
        print("[decode] no decoded files found -- using raw labels throughout")
        return ce

    DEC = {"success_strict": "decoded_strict",
           "success_lenient": "decoded_lenient"}
    dec_col = DEC.get(metric)
    if dec_col is None:
        print(f"[decode] metric {metric!r} has no decoded counterpart (the "
              f"decode pass ran the two classifiers only, not the heuristic "
              f"tier) -- using raw labels throughout")
        return ce

    ce = ce.copy()
    ce["_dec"] = pd.NA
    ce["_enc"] = False
    total = 0

    for f in files:
        d = pd.read_csv(f, low_memory=False)
        if "is_encoded" not in d.columns or dec_col not in d.columns:
            continue
        d = d[d.is_encoded.fillna(False)]
        if d.empty:
            continue

                # NEITHER instruction_idx NOR custom_id is model-unique:
        # instruction_idx is "{framing}|{template}|{order}|{idx}" and custom_id
        # is "r{i}", so the same value exists in every model's file. Merging on
        # them alone applies one model's decoded labels to all the others.
        if "instruction_idx" in d.columns and "instruction_idx" in ce.columns:
            key = ["model", "instruction_idx"]
        elif "custom_id" in d.columns and "custom_id" in ce.columns:
            key = ["model", "custom_id"]
        else:
            # Composite fallback. Stage B rows have `cell`; Stage A rows have
            # `framing` and no cell column.
            unit = ("cell" if "cell" in ce.columns
                    else "framing" if "framing" in ce.columns
                    else "template")
            key = ["model", unit, "instruction_id"]

        sub = d[key + [dec_col]].rename(columns={dec_col: "_new"})
        sub = sub.drop_duplicates(subset=key)
        ce = ce.merge(sub, on=key, how="left")
        hit = ce["_new"].notna()
        ce.loc[hit, "_dec"] = ce.loc[hit, "_new"]
        ce.loc[hit, "_enc"] = True
        total += int(hit.sum())
        ce = ce.drop(columns=["_new"])

    if total == 0:
        print("[decode] decoded files present but nothing matched -- "
              "check the merge keys")
        return ce.drop(columns=["_dec", "_enc"])

    before = ce[metric].fillna(False).mean()
    ce.loc[ce["_enc"], metric] = ce.loc[ce["_enc"], "_dec"]
    after = ce[metric].fillna(False).mean()

    print(f"[decode] {total:,} encoded responses relabelled "
          f"({total/len(ce):.1%} of rows in this stage)")
    print(f"[decode] pooled {metric}: {before:.1%} raw -> {after:.1%} decoded")

    return ce.drop(columns=["_dec", "_enc"])


def build_space(metric: str, use_decoded: bool = True) -> pd.DataFrame:
    """
    One row per (model, configuration) with its ASR and n.

    kind is framing_only | template_only | composed, so downstream analysis
    can ask "did the winner compose?" without string-parsing config names.
    """
    fr = _load(["results/stage_a/stage_a_*.csv"], "framing", metric)
    te = _load(["results/stage_a_templates/stage_a*_*.csv"], "template", metric)
    ce = _load(["results/stage_b/checkpoints/exp20_*.csv",
                "results/stage_b/stage_b_*.csv"], "cell", metric)

    # Stage B only searches the 11 selected framings and 4 selected templates.
    # Stage A screened 43 and Stage A-T screened 12, so restrict the
    # single-component configurations to the ones actually in the design space
    # -- otherwise c* could be a framing the algorithm can never propose.
    from run_stage_b import STAGE_B_FRAMINGS
    from attacks.libra_framings import UTOS_TOP4
    fr = fr[fr.framing.isin(STAGE_B_FRAMINGS)]
    te = te[te.template.isin(UTOS_TOP4)]

    # AdvBench ids 337 and 51 are permanently rejected by the GPT
    # pre-generation filter (HTTP 400, "possible biological risk"). Some cells
    # got through before the exclusion was added at submit time, so without
    # this a handful of cells are measured on 180 instructions and the rest on
    # 178. Make the GPT family uniformly 178 and state that in the methods.
    GPT_BLOCKED = {337, 51}
    fr, te, ce = [
        d if d.empty else
        d[~(d.model.str.startswith("gpt-5") & d.instruction_id.isin(GPT_BLOCKED))]
        for d in (fr, te, ce)
    ]

        # Correct BOTH the composed rows and the framing-only rows. Correcting only
    # the composed ones leaves plinius_l33tspeak standalone at its raw,
    # inflated value -- which then becomes c* for gpt-5-mini and gpt-5-nano
    # even though decoding collapses those same responses to ~0% in the
    # composed cells. Templates need no correction: no template forces encoding.
    if use_decoded:
        if not ce.empty:
            ce = _apply_decoded(ce, metric)
        if not fr.empty:
            fr = _apply_decoded(fr, metric,
                                "results/stage_a_decoded/decoded_*.csv")

    rows = []
    for d, key, kind in ((fr, "framing", "framing_only"),
                         (te, "template", "template_only"),
                         (ce, "cell", "composed")):
        if d.empty:
            continue
        g = d.groupby(["model", key])[metric].agg(["mean", "size"]).reset_index()
        g.columns = ["model", "config", "asr", "n"]
        g["kind"] = kind
        # framing / template columns for the composed rows, so interference
        # can be decomposed by component.
        if kind == "composed":
            parts = g.config.str.split("|", expand=True)
            g["framing"], g["template"], g["order"] = parts[0], parts[1], parts[2]
        elif kind == "framing_only":
            g["framing"], g["template"], g["order"] = g.config, None, None
        else:
            g["framing"], g["template"], g["order"] = None, g.config, None
        rows.append(g)

    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="success_strict",
                    choices=["success_strict", "jailbreak_success_ensemble",
                             "success_lenient"])
    ap.add_argument("--min-n", type=int, default=150,
                    help="drop configurations measured on fewer instructions "
                         "than this (incomplete runs)")
    ap.add_argument("--no-decode", action="store_true",
                    help="use raw judge labels even for encoded responses "
                         "(reproduces the pre-correction numbers)")
    ap.add_argument("--out", default="results/search_space.csv")
    args = ap.parse_args()

    sp = build_space(args.metric, use_decoded=not args.no_decode)
    if sp.empty:
        raise SystemExit("no data found")

    incomplete = sp[sp.n < args.min_n]
    if len(incomplete):
        print(f"[warn] dropping {len(incomplete)} configuration(s) with n < {args.min_n}")
        for m, k in incomplete.groupby(["model", "kind"]).size().items():
            print(f"       {m[0]:<24} {m[1]:<14} {k}")
        sp = sp[sp.n >= args.min_n]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sp.to_csv(args.out, index=False)
    print(f"metric: {args.metric}\nwrote {args.out}  ({len(sp)} rows)\n")

    # ---- 1. the search space and c* per model ----------------------------
    print("=" * 78)
    print("SEARCH SPACE AND GROUND TRUTH")
    print("=" * 78)
    print(f"{'model':<24} {'configs':>8} {'c* ASR':>8}  c*")
    for m in sorted(sp.model.unique()):
        x = sp[sp.model == m]
        best = x.loc[x.asr.idxmax()]
        print(f"{m:<24} {len(x):>8} {best.asr:>8.1%}  [{best.kind}] {best.config}")

    print(f"\n{'model':<24} " + "  ".join(f"{k:>14}" for k in
          ("framing_only", "template_only", "composed")))
    for m in sorted(sp.model.unique()):
        x = sp[sp.model == m]
        cells = []
        for k in ("framing_only", "template_only", "composed"):
            y = x[x.kind == k]
            cells.append(f"{y.asr.max():.1%} ({len(y)})" if len(y) else "-")
        print(f"{m:<24} " + "  ".join(f"{c:>14}" for c in cells))

    # ---- 2. does composition help? ---------------------------------------
    print("\n" + "=" * 78)
    print("SYNERGY: best composed cell vs best single component")
    print("=" * 78)
    syn = []
    for m in sorted(sp.model.unique()):
        x = sp[sp.model == m]
        comp = x[x.kind != "composed"]
        cell = x[x.kind == "composed"]
        if comp.empty or cell.empty:
            continue
        bc, bx = comp.asr.max(), cell.asr.max()
        n_better = int((cell.asr > bc).sum())
        syn.append({"model": m, "best_component": bc, "best_cell": bx,
                    "synergy": bx - bc, "n_cells_better": n_better,
                    "n_cells": len(cell)})
    syn = pd.DataFrame(syn)
    print(f"{'model':<24} {'best comp':>10} {'best cell':>10} {'synergy':>9} "
          f"{'cells >':>9}")
    for r in syn.itertuples():
        print(f"{r.model:<24} {r.best_component:>10.1%} {r.best_cell:>10.1%} "
              f"{r.synergy:>+9.1%} {r.n_cells_better:>4}/{r.n_cells}")

    if len(syn) >= 3:
        rho = syn.best_component.corr(syn.synergy, method="spearman")
        print(f"\nSpearman(best single component, synergy) = {rho:+.3f}")
        print("Negative means composition helps only where no single component "
              "already works.\nWith <6 models this is indicative, not a test -- "
              "report it as a pattern with\nthe per-model numbers, not as a "
              "correlation coefficient on its own.")

    # ---- 3. where does the interference come from? -----------------------
    print("\n" + "=" * 78)
    print("INTERFERENCE BY FRAMING: standalone vs its own composed cells")
    print("=" * 78)
    for m in sorted(sp.model.unique()):
        x = sp[sp.model == m]
        alone = x[x.kind == "framing_only"].set_index("framing").asr
        cells = x[x.kind == "composed"]
        if alone.empty or cells.empty:
            continue
        agg = cells.groupby("framing").asr.agg(["max", "mean"])
        j = agg.join(alone.rename("alone"), how="inner").dropna()
        if j.empty:
            continue
        j["delta_best"] = j["max"] - j["alone"]
        j = j.sort_values("alone", ascending=False)
        print(f"\n{m}")
        print(f"  {'framing':<26} {'alone':>8} {'best cell':>10} "
              f"{'mean cell':>10} {'delta':>8}")
        for f, r in j.iterrows():
            print(f"  {f:<26} {r['alone']:>8.1%} {r['max']:>10.1%} "
                  f"{r['mean']:>10.1%} {r['delta_best']:>+8.1%}")
        hurt = int((j.delta_best < 0).sum())
        print(f"  framings where composing never beats the framing alone: "
              f"{hurt}/{len(j)}")

    # ---- 4. order ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("COMPOSITION ORDER")
    print("=" * 78)
    cells = sp[sp.kind == "composed"]
    if not cells.empty:
        piv = cells.pivot_table(index="model", columns="order", values="asr")
        if piv.shape[1] == 2:
            a, b = piv.columns
            piv["delta"] = piv[a] - piv[b]
            print(piv.applymap(lambda v: f"{v:+.1%}" if pd.notna(v) else "-")
                  .to_string())
            print(f"\n(delta = {a} minus {b}; sign flipping across models means "
                  f"order must stay\nin the search space rather than being fixed "
                  f"to whichever wins on average.)")


if __name__ == "__main__":
    main()