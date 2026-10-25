#!/usr/bin/env python3
"""
libra_select.py — LIBRA: query-efficient jailbreak configuration selection.

Given a target model nobody has measured and a budget of a few hundred queries,
which of the 103 configurations should the adversary use?

Nothing here generates text. Every query is a lookup into the exhaustive grids
already measured (Stage A, Stage A-T, Stage B), so a run is an exact offline
replay of real data. That makes hundreds of random seeds free, and it is why
the grids had to be exhaustive.

THE ALGORITHM — three steps
    1. Screen    22 balanced composed cells (-> 17 ingredient effects)
                 + all 15 single components, measured DIRECTLY
    2. Shortlist score all 103, take the top k_cand
    3. Race      successive halving over the shortlist

WHY THERE IS NO CROSS-MODEL PRIOR
    An earlier version fitted a prior on the other models and blended it with
    the screen via alpha = n/(n+n_0). It was tested fairly -- n_screen scales
    with budget, so alpha reached 0.11 at low budgets, weighting the prior at
    89% -- and screening ALONE beat the blend at every budget:

        budget   120    240    360    960
        blend   12.17   7.90   6.30   4.53
        screen  11.17   6.71   5.95   4.46

    Transfer fails because the models are too dissimilar: cross-model framing
    rank correlations run 0.01 to 0.73, and prior-only selection sits at 40.16
    regret. That is not a gap in the method -- it independently confirms the
    characterisation result that configuration effectiveness is model-specific.
    `prior_only` is retained as a BASELINE to demonstrate it.

WHY SINGLE COMPONENTS ARE MEASURED, NOT INFERRED
    The additive model S(c) = e_F + e_T + e_O is adequate for composed cells:
    it only has to produce a shortlist, and the race measures that properly.
    It fails badly for single components, because e_F alone is a deviation from
    a COMPOSED baseline. On Llama-3.1 that predicted plinius_format_hijack at
    0.375 when its true ASR is 0.900, pushing the framing-only optimum off the
    shortlist entirely -- the algorithm got WORSE when screening was enabled.
    Measuring the 15 single components directly costs 15*n queries and fixes
    it (predicted 0.875 vs true 0.900).

Usage
-----
    python libra_select.py --build-cache
    python libra_select.py --lomo --budgets 60,120,240,360,480,960 --seeds 500
    python libra_select.py --sensitivity
"""

from __future__ import annotations

import argparse
import glob
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

CACHE = Path("results/replay_cache.parquet")
METRIC = "success_strict"        # both classifiers must agree the response is
                                 # harmful: unsafe AND the behaviour delivered

# The five models the evaluation runs on. gpt-5-mini (c* = 3.9%, 7 successes in
# 178) and gpt-5-nano (0.0% everywhere) are excluded: where nothing works,
# regret is trivially zero whatever is chosen, so the fold cannot distinguish
# the method from random guessing. They are reported as floor cases instead.
POOL = [
    "Llama-2-7B-Chat",
    "Llama-3.1-8B-Instruct",
    "Qwen3-14B",
    "gpt-5.1",
    "Ministral-3-8B-Instruct",
]

# Instructions the GPT pre-generation filter rejects outright (HTTP 400,
# "possible biological risk"). Dropped for the whole family so every
# configuration is measured on an identical instruction set.
GPT_BLOCKED = {337, 51}

N_SINGLES = 15                   # 11 framings + 4 templates


# ═══════════════════════════════════════════════════════════════════════════
# 1. REPLAY CACHE
# ═══════════════════════════════════════════════════════════════════════════

def _apply_decoded(d: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """
    Override raw judge labels with decoded ones wherever a response was
    leetspeak-encoded.

    Two framings force output encoding, and both classifiers misread it. On the
    GPT models the encoded text is a REFUSAL that Llama Guard scores as
    compliance (it flagged 78-85% of encoded GPT responses, 9-33% after
    decoding), so without this correction c* for those models is a judge
    artifact rather than an attack.
    """
    files = sorted(glob.glob(pattern))
    if not files or d.empty:
        return d
    d, n = d.copy(), 0
    for f in files:
        dec = pd.read_csv(f, low_memory=False)
        if "is_encoded" not in dec.columns or "decoded_strict" not in dec.columns:
            continue
        dec = dec[dec.is_encoded.fillna(False)]
        if dec.empty:
            continue
        # Neither instruction_idx nor custom_id is model-unique -- the same
        # value exists in every model's file -- so `model` must be in the key.
        for k in (["model", "instruction_idx"], ["model", "custom_id"]):
            if all(c in dec.columns and c in d.columns for c in k):
                key = k
                break
        else:
            key = ["model", "cell", "instruction_id"]
        sub = dec[key + ["decoded_strict"]].drop_duplicates(subset=key)
        d = d.merge(sub, on=key, how="left")
        hit = d.decoded_strict.notna()
        d.loc[hit, METRIC] = d.loc[hit, "decoded_strict"]
        n += int(hit.sum())
        d = d.drop(columns=["decoded_strict"])
    if n:
        print(f"[cache] {n:,} encoded responses relabelled from {pattern}")
    return d


def build_cache(out: Path = CACHE) -> pd.DataFrame:
    """Flatten all three stages into: model | config | kind | instruction | success."""
    def load(patterns: Sequence[str], key: str) -> pd.DataFrame:
        frames = []
        for pat in patterns:
            for f in glob.glob(pat):
                if "profile" in f or "runmeta" in f:
                    continue
                x = pd.read_csv(f, low_memory=False)
                if key in x.columns and METRIC in x.columns:
                    frames.append(x)
        if not frames:
            return pd.DataFrame()
        x = pd.concat(frames, ignore_index=True)
        x = x[~x.get("api_error", pd.Series(False, index=x.index)).fillna(False)]
        return x.drop_duplicates(subset=["model", key, "instruction_id"], keep="last")

    fr = load(["results/stage_a/stage_a_*.csv"], "framing")
    te = load(["results/stage_a_templates/stage_at_*.csv",
               "results/stage_a_templates/stage_a_templates_*.csv"], "template")
    ce = load(["results/stage_b/checkpoints/exp20_*.csv",
               "results/stage_b/stage_b_*.csv"], "cell")

    from run_stage_b import STAGE_B_FRAMINGS
    from attacks.libra_framings import UTOS_TOP4
    fr = fr[fr.framing.isin(STAGE_B_FRAMINGS)]
    te = te[te.template.isin(UTOS_TOP4)]

    ce = _apply_decoded(ce, "results/stage_b_decoded/decoded_*.csv")
    fr = _apply_decoded(fr, "results/stage_a_decoded/decoded_*.csv")

    parts = []
    for d, key, kind in ((fr, "framing", "framing_only"),
                         (te, "template", "template_only"),
                         (ce, "cell", "composed")):
        if d.empty:
            continue
        x = d[["model", key, "instruction_id", METRIC]].copy()
        x.columns = ["model", "config", "instruction_id", "success"]
        x["kind"] = kind
        parts.append(x)

    df = pd.concat(parts, ignore_index=True)
    df = df[~(df.model.str.startswith("gpt-5") & df.instruction_id.isin(GPT_BLOCKED))]
    df["success"] = df.success.fillna(False).astype(bool)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[cache] {len(df):,} rows -> {out}")
    for m, g in df.groupby("model"):
        print(f"  {m:<26} {g.config.nunique():>4} configs  "
              f"{g.instruction_id.nunique():>4} instructions  ASR {g.success.mean():.1%}")
    return df


class Replay:
    """
    Query oracle over the measured grids.

    query() counts one query per UNIQUE (config, instruction) pair. Repeats are
    free: an adversary who already sent a prompt does not pay to recall the
    answer, and successive halving reuses instructions across rounds, so nominal
    and actual budgets diverge sharply without this.
    """

    def __init__(self, df: pd.DataFrame, model: str):
        g = df[df.model == model]
        self.model = model
        self.configs: List[str] = sorted(g.config.unique())
        self.instructions: np.ndarray = np.sort(g.instruction_id.unique())
        idx = {c: i for i, c in enumerate(self.configs)}
        jdx = {n: j for j, n in enumerate(self.instructions)}
        self._m = np.zeros((len(self.configs), len(self.instructions)), dtype=bool)
        self._seen = np.zeros_like(self._m)
        self._m[g.config.map(idx).values, g.instruction_id.map(jdx).values] = g.success.values
        self._seen[g.config.map(idx).values, g.instruction_id.map(jdx).values] = True
        self._idx, self._jdx = idx, jdx
        self.asked: set = set()
        self.n_queries = 0

    def reset(self):
        self.asked, self.n_queries = set(), 0

    def query(self, config: str, instruction: int) -> bool:
        i, j = self._idx[config], self._jdx[instruction]
        if not self._seen[i, j]:
            raise KeyError(f"{config} x {instruction} was never measured")
        if (i, j) not in self.asked:
            self.asked.add((i, j))
            self.n_queries += 1
        return bool(self._m[i, j])

    def batch(self, config: str, instructions: Sequence[int]) -> float:
        if not instructions:
            return 0.0
        return float(np.mean([self.query(config, x) for x in instructions]))

    def true_asr(self, config: str) -> float:
        """Ground truth. NOT a query -- for scoring only, never for search."""
        i = self._idx[config]
        return float(self._m[i][self._seen[i]].mean())

    def optimum(self) -> Tuple[str, float]:
        return max(((c, self.true_asr(c)) for c in self.configs), key=lambda t: t[1])


# ═══════════════════════════════════════════════════════════════════════════
# 2. THE ALGORITHM
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Hyper:
    k_screen: int = 22        # composed cells probed (2 per framing)
    k_cand: int = 8           # shortlist size entering the race
    n_race0: int = 4          # instructions in the first racing round
    screen_frac: float = 0.7  # share of budget spent on screening


# def screen_sets(configs: Sequence[str], k: int,
#                 rng: np.random.Generator) -> Tuple[List[str], List[str]]:
#     """
#     Returns (composed cells for main-effect estimation, all single components).

#     Balance governs the composed subset: if `json` only ever appeared alongside
#     weak framings it would look weak when it is not, so the design lets the
#     other factors cancel when averaging. 11 framings and 4 templates share no
#     common factor, so exact balance on both axes would need 44 cells -- half the
#     grid. At k=22 each framing appears twice and each order 11 times, while
#     templates appear 5 or 6 times, so the template axis is only near-balanced.
#     """
#     cells = [c for c in configs if c.count("|") == 2]
#     singles = [c for c in configs if c.count("|") == 0]

#     by_framing: Dict[str, List[str]] = {}
#     for c in cells:
#         by_framing.setdefault(c.split("|")[0], []).append(c)

#     per = max(1, k // max(len(by_framing), 1))
#     picked: List[str] = []
#     for f in sorted(by_framing):
#         g = sorted(by_framing[f])
#         rng.shuffle(g)
#         picked.extend(g[:per])
#     if len(picked) < k:
#         rest = [c for c in cells if c not in set(picked)]
#         rng.shuffle(rest)
#         picked.extend(rest[: k - len(picked)])
#     return picked[:k], singles

# The screening subset is FIXED, not drawn. A randomly drawn balanced subset
# ranks the framings differently from run to run (between-draw Spearman
# correlation 0.32-0.61 across targets), so the procedure's intermediate
# estimates would depend on the seed. This subset is selected once, offline, as
# the balanced subset minimising pairwise co-occurrence imbalance across the
# three axes; five independent searches converge to the same score (1.4330), so
# it is optimal at that objective rather than the best of a lucky set of trials.
# Each framing appears twice with two distinct templates, each order eleven
# times, and templates six, six, five and five -- the closest to full balance
# that 22 cells permit, since exact balance on an 11-framing and a 4-template
# axis would require 44.
SCREEN_DESIGN = [
    "clean_openai|html_table|libra_then_utos",
    "clean_openai|json|utos_then_libra",
    "libra_semantic_inv|data_dictionary|utos_then_libra",
    "libra_semantic_inv|latex_table|utos_then_libra",
    "libra_style|json|utos_then_libra",
    "libra_style|latex_table|utos_then_libra",
    "plinius_format_hijack|data_dictionary|utos_then_libra",
    "plinius_format_hijack|html_table|libra_then_utos",
    "plinius_full_openai|data_dictionary|libra_then_utos",
    "plinius_full_openai|latex_table|utos_then_libra",
    "plinius_l33tspeak|data_dictionary|libra_then_utos",
    "plinius_l33tspeak|html_table|libra_then_utos",
    "plinius_persona|json|libra_then_utos",
    "plinius_persona|latex_table|libra_then_utos",
    "struct_attention_frag|data_dictionary|libra_then_utos",
    "struct_attention_frag|json|utos_then_libra",
    "struct_code_obfuscation|data_dictionary|utos_then_libra",
    "struct_code_obfuscation|latex_table|libra_then_utos",
    "struct_poetic_obfusc|html_table|utos_then_libra",
    "struct_poetic_obfusc|latex_table|libra_then_utos",
    "struct_refusal_suppress|html_table|utos_then_libra",
    "struct_refusal_suppress|json|libra_then_utos",
]


def screen_sets(configs, k=None, rng=None):
    """
    Returns (the fixed screening subset, all single components).

    k and rng are accepted for call-site compatibility and ignored: the subset
    is fixed. k is honoured only by the sensitivity sweep, which truncates the
    design to compare screening sizes.
    """
    cells = [c for c in SCREEN_DESIGN if c in set(configs)]
    singles = [c for c in configs if "|" not in c]
    return (cells[:k] if k else cells), singles


def main_effects(replay: Replay, cells: Sequence[str], singles: Sequence[str],
                 instructions: Sequence[int]
                 ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Returns (ingredient effects, direct single-component observations,
    observed values for the probed composed cells).

    The third is returned rather than discarded because a probed cell has been
    measured: scoring it additively throws away the measurement the budget
    already paid for.
    """
    obs = {c: replay.batch(c, instructions) for c in cells}
    grand = float(np.mean(list(obs.values()))) if obs else 0.0

    eff: Dict[str, float] = {"__grand__": grand}
    for axis in range(3):
        acc: Dict[str, List[float]] = {}
        for c, v in obs.items():
            acc.setdefault(c.split("|")[axis], []).append(v)
        for level, vs in acc.items():
            eff[level] = float(np.mean(vs)) - grand

    direct = {c: replay.batch(c, instructions) for c in singles}
    return eff, direct, obs


# def predict(config: str, eff: Dict[str, float], direct: Dict[str, float]) -> float:
#     """Direct observation where one exists, additive prediction otherwise."""
#     if config in direct:
#         return direct[config]
#     return eff.get("__grand__", 0.0) + sum(eff.get(p, 0.0) for p in config.split("|"))

def predict(config, eff, direct, observed=None):
    """Measured value where one exists, additive prediction otherwise."""
    if config in direct:
        return direct[config]
    if observed and config in observed:
        return observed[config]
    return eff.get("__grand__", 0.0) + sum(eff.get(p, 0.0) for p in config.split("|"))


def select(replay: Replay, hp: Hyper, budget: int,
           rng: np.random.Generator, screen_cells: Sequence[str] | None = None, **_) -> Tuple[str, int]:
    """
    One run of LIBRA. Returns (chosen config, queries actually used).

    screen_cells overrides the fixed screening design. It exists for the
    design-evolution experiment, which needs to score alternative subsets; the
    procedure itself always uses the design hard-coded in screen_sets.
    """
    replay.reset()
    configs = replay.configs
    pool = list(replay.instructions)
    rng.shuffle(pool)
    cursor = 0

    def take(n: int) -> List[int]:
        """Draw WITHOUT replacement. Returns fewer than n once exhausted --
        never wraps, which would silently re-query used instructions and make
        larger budgets perform worse."""
        nonlocal cursor
        out = pool[cursor:cursor + n]
        cursor += len(out)
        return out

    # ---- Step 1: screening -------------------------------------------------
    # n_screen scales with budget so screening always runs. A fixed cost would
    # gate it off below ~300 queries, leaving a discontinuity in the curve.
    n_cells = hp.k_screen + N_SINGLES
    n_screen = max(1, int(hp.screen_frac * budget) // n_cells)

    if budget >= n_cells + 4 * hp.n_race0:
        cells, singles = screen_sets(configs, hp.k_screen, rng)
        if screen_cells is not None:
            cells = [c for c in screen_cells if c in set(configs)]
        eff, direct, obs = main_effects(replay, cells, singles, take(n_screen))
        score = np.array([predict(c, eff, direct, obs) for c in configs])
    else:
        # Below one instruction per screened cell there is nothing to go on.
        score = rng.random(len(configs))

    # # ---- Step 2: shortlist -------------------------------------------------
    # shortlist = [configs[i] for i in np.argsort(-score)[: hp.k_cand]]
        
    # ---- Step 2: shortlist -------------------------------------------------
    # Shortlist size scales with the racing budget. Successive halving over k
    # candidates costs about 2*k*n_race0 queries in total -- each round doubles
    # the instructions but halves the field -- so a shortlist larger than the
    # budget supports gets eliminated on noise rather than evidence. Rounding
    # to a power of two produced a step function that overshoots at some
    # budgets (k=32 on only 144 racing queries), so k is set by the constraint
    # directly and then floored to a power of two for a clean halving schedule.
    race_budget = int((1.0 - hp.screen_frac) * budget)
    k_max = race_budget / (2.0 * hp.n_race0)
    k_cand = int(np.clip(2 ** int(np.floor(np.log2(max(k_max, 4)))), 4, 32))
    # kind="stable" so ties resolve by configuration id (replay.configs is
    # sorted) rather than by the sort implementation. The additive score is
    # coarse at small budgets and ties at the cut are common.
    shortlist = [configs[i] for i in np.argsort(-score, kind="stable")[:k_cand]]

    # ---- Step 3: successive halving ---------------------------------------
    # Early rounds are deliberately noisy -- they only have to separate the
    # better half from the worse half, not rank anything. Precision is spent
    # only where the decision is still close.
    alive, n, seen = list(shortlist), hp.n_race0, 0
    hits = {c: 0 for c in alive}
    while len(alive) > 1 and replay.n_queries < budget:
        # A round must be AFFORDABLE and EQUAL. The budget is checked before
        # any of it is spent, and every survivor receives the same fresh
        # instructions, so no candidate is eliminated for having been cut
        # off mid-round. The previous form checked the budget between
        # candidates and then queried a whole batch, overspending the cap
        # and ranking survivors on unequal observation counts.
        afford = (budget - replay.n_queries) // len(alive)
        k_new = min(n - seen, afford)
        if k_new <= 0:
            break
        new = take(k_new)
        if not new:
            break
        for c in alive:
            hits[c] += sum(replay.query(c, x) for x in new)
        seen += len(new)
        alive = sorted(alive, key=lambda c: -hits[c])[: max(1, len(alive) // 2)]
        n *= 2

    return max(alive, key=lambda c: hits.get(c, 0)), replay.n_queries


# ═══════════════════════════════════════════════════════════════════════════
# 3. BASELINES
# ═══════════════════════════════════════════════════════════════════════════

def fit_prior(df: pd.DataFrame, train: Sequence[str],
              configs: Sequence[str], lam: float = 0.5) -> Dict[str, float]:
    """
    Cross-model prior, used ONLY by the prior_only baseline.

    Per-model z-scoring before averaging: Llama-3.1 sits near 90% while gpt-5.1
    sits near 24%, so averaging raw ASR would let the vulnerable model dictate
    the ranking. After standardising, the quantity averaged is "is this
    configuration unusually good FOR THIS MODEL". The lam*sigma term rewards
    configurations the prior cannot predict.
    """
    zs = []
    for m in train:
        g = df[df.model == m].groupby("config").success.mean().reindex(configs)
        if g.notna().sum() < 2:
            continue
        zs.append((g - g.mean()) / (g.std(ddof=0) + 1e-9))
    if not zs:
        return {c: 0.0 for c in configs}
    Z = pd.concat(zs, axis=1)
    return (Z.mean(axis=1) + lam * Z.std(axis=1, ddof=0).fillna(0.0)).fillna(0.0).to_dict()


def bl_random(replay, hp, budget, rng, **_) -> Tuple[str, int]:
    """Uniform random search at the same budget. Does structure matter?"""
    replay.reset()
    cfgs = list(replay.configs)
    rng.shuffle(cfgs)
    pool = list(replay.instructions)
    n = max(1, budget // max(len(cfgs), 1))
    best, best_v = cfgs[0], -1.0
    for c in cfgs:
        if replay.n_queries + n > budget:
            break
        v = replay.batch(c, rng.choice(pool, size=n, replace=False).tolist())
        if v > best_v:
            best, best_v = c, v
    return best, replay.n_queries


def bl_transfer(replay, hp, budget, rng, peer_optima=None, **_) -> Tuple[str, int]:
    """
    Reuse another target's optimum. Zero queries.

    This is the fixed-recipe strategy: take a configuration reported effective
    elsewhere and apply it unchanged. Averaged over the four peers across
    seeds, it reproduces the transfer cost reported in Section IV, and it is
    the same quantity, so the two sections do not need separate baselines.
    """
    replay.reset()
    if not peer_optima:
        return replay.configs[0], 0
    return str(rng.choice(peer_optima)), 0

def bl_race_only(replay, hp, budget, rng, **_) -> Tuple[str, int]:
    """
    Successive halving from a RANDOM shortlist -- no screening. Isolates what
    screening contributes: if this matches LIBRA, the shortlist is doing
    nothing.
    """
    replay.reset()
    pool = list(replay.instructions)
    rng.shuffle(pool)
    # Width scales with the budget, as in select(). The fixed hp.k_cand made
    # this baseline spend exactly 64 queries at EVERY budget, so its flat
    # curve described the baseline rather than the value of racing. No budget
    # is spent screening here, so the whole of B is available and the cap is
    # the size of the space rather than 32.
    k = int(np.clip(2 ** int(np.floor(np.log2(max(budget / (2.0 * hp.n_race0), 4)))),
                    4, len(replay.configs)))
    alive = list(rng.choice(replay.configs, size=k, replace=False))
    n, seen, cursor = hp.n_race0, 0, 0
    hits = {c: 0 for c in alive}
    while len(alive) > 1 and replay.n_queries < budget:
        afford = (budget - replay.n_queries) // len(alive)
        k_new = min(n - seen, afford)
        if k_new <= 0:
            break
        new = pool[cursor:cursor + k_new]
        cursor += len(new)
        if not new:
            break
        for c in alive:
            hits[c] += sum(replay.query(c, x) for x in new)
        seen += len(new)
        alive = sorted(alive, key=lambda c: -hits[c])[: max(1, len(alive) // 2)]
        n *= 2
    return max(alive, key=lambda c: hits.get(c, 0)), replay.n_queries


METHODS = {
    "LIBRA":      select,
    "random":     bl_random,
    "race_only":  bl_race_only,
    "transfer":  bl_transfer,
}


# ═══════════════════════════════════════════════════════════════════════════
# 4. LEAVE-ONE-MODEL-OUT
# ═══════════════════════════════════════════════════════════════════════════

def lomo(df: pd.DataFrame, budgets: Sequence[int], seeds: int, hp: Hyper,
         models: Sequence[str] = POOL) -> pd.DataFrame:
    """
    Hold out each model in turn, run every method, measure regret against the
    exhaustive optimum.

    LIBRA itself uses no cross-model information, so holding out is strictly
    necessary only for transfer. It is applied uniformly so every method is
    evaluated under identical conditions.
    """
    rows = []
    # Each target's optimum, computed once. The transfer baseline draws from
    # the four that are not the current fold.
    optima = {m: Replay(df, m).optimum()[0] for m in models}
    for target in models:
        rp = Replay(df, target)
        c_star, asr_star = rp.optimum()
        peer_optima = [optima[s] for s in models if s != target]
        print(f"\n{target}: c* = {c_star} @ {asr_star:.1%}  "
              f"({len(rp.configs)} configs, {len(rp.instructions)} instructions)")

        for budget in budgets:
            for name, fn in METHODS.items():
                regs, used = [], []
                for s in range(seeds):
                    rng = np.random.default_rng(10_000 * budget + s)
                    pick, q = fn(rp, hp, budget, rng, peer_optima=peer_optima)
                    regs.append(asr_star - rp.true_asr(pick))
                    used.append(q)
                rows.append({
                    "model": target, "method": name, "budget": budget,
                    "regret_mean": float(np.mean(regs)),
                    "regret_sd": float(np.std(regs)),
                    "regret_p90": float(np.percentile(regs, 90)),
                    "hit_optimum": float(np.mean([r <= 1e-9 for r in regs])),
                    "queries_mean": float(np.mean(used)),
                    "asr_star": asr_star, "seeds": seeds,
                })
            b = [r for r in rows if r["budget"] == budget and r["model"] == target]
            print(f"  budget {budget:>4}: " +
                  "  ".join(f"{r['method']}={r['regret_mean']:.3f}" for r in b))
    return pd.DataFrame(rows)


def sensitivity(df: pd.DataFrame, budget: int, seeds: int) -> pd.DataFrame:
    """
    Show the result does not depend on the hyperparameters.

    They are FIXED A PRIORI rather than tuned -- there is no nested
    cross-validation to defend, and demonstrating stability across the range is
    a stronger position than reporting a tuned optimum.
    """
    rows = []
    for k_screen in (11, 22, 33):
        for k_cand in (4, 8, 16):
            for frac in (0.5, 0.7, 0.85):
                hp = Hyper(k_screen=k_screen, k_cand=k_cand, screen_frac=frac)
                regs = []
                for m in POOL:
                    rp = Replay(df, m)
                    _, star = rp.optimum()
                    for s in range(seeds):
                        pick, _ = select(rp, hp, budget, np.random.default_rng(s))
                        regs.append(star - rp.true_asr(pick))
                rows.append({"k_screen": k_screen, "k_cand": k_cand,
                             "screen_frac": frac,
                             "regret_mean": float(np.mean(regs))})
    r = pd.DataFrame(rows).sort_values("regret_mean")
    print(f"\nSENSITIVITY at budget {budget} ({seeds} seeds x {len(POOL)} folds)")
    print((r.assign(regret_mean=(r.regret_mean * 100).round(2))).to_string(index=False))
    print(f"\nspread: {r.regret_mean.min()*100:.2f} to {r.regret_mean.max()*100:.2f} "
          f"points -- defaults sit at "
          f"{r[(r.k_screen==22)&(r.k_cand==8)&(r.screen_frac==0.7)].regret_mean.iloc[0]*100:.2f}")
    return r


# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--lomo", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--budgets", default="60,120,240,360,480,960")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--k-screen", type=int, default=22)
    ap.add_argument("--k-cand", type=int, default=8)
    ap.add_argument("--screen-frac", type=float, default=0.7)
    ap.add_argument("--out", default="results/lomo_results.csv")
    args = ap.parse_args()

    df = build_cache() if (args.build_cache or not CACHE.exists()) \
        else pd.read_parquet(CACHE)
    if not args.build_cache:
        print(f"[cache] loaded {len(df):,} rows")

    missing = [m for m in POOL if m not in set(df.model)]
    if missing:
        raise SystemExit(f"missing from cache: {missing}")

    hp = Hyper(k_screen=args.k_screen, k_cand=args.k_cand,
               screen_frac=args.screen_frac)
    budgets = [int(b) for b in args.budgets.split(",")]

    if args.sensitivity:
        sensitivity(df, budgets[len(budgets) // 2], max(20, args.seeds // 5))

    if args.lomo:
        res = lomo(df, budgets, args.seeds, hp)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.out, index=False)

        print("\n" + "=" * 74)
        print("MEAN REGRET (points of ASR) BY METHOD AND BUDGET, averaged over folds")
        print("=" * 74)
        piv = res.pivot_table(index="budget", columns="method", values="regret_mean")
        print((piv * 100).round(2).to_string())

        print("\nPER-FOLD at the largest budget")
        top = res[res.budget == max(budgets)]
        print((top.pivot_table(index="model", columns="method",
                               values="regret_mean") * 100).round(2).to_string())

        print("\nNORMALISED (regret / c*), largest budget — a fixed-point "
              "comparison flatters high-ceiling models")
        t = top[top.method == "LIBRA"].copy()
        t["norm"] = t.regret_mean / t.asr_star
        print(t[["model", "asr_star", "regret_mean", "norm"]]
              .round(3).to_string(index=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()