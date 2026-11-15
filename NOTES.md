# Notes

Caveats, negative results and known limits of this release. Each item is
reported as it came out rather than as we would have liked it.

---

## What cannot be reproduced here

Most of the paper regenerates from the released caches. Two things do not.

**The held-out grid (Sections IV-D and V-D).** `analyse_eval_half.py` and
`heldout_replay.py` both read `results/eval_half/*.csv`, the per-cell outcomes
of all 103 configurations on all 340 evaluation instructions for the four
open-weight targets, 140,080 pairs. Those per-cell files are **not released**,
by the same rule that governs the raw generations. What is released is the
derived output: `results/eval_half/heldout_103.csv` (four summary rows) and
`results/heldout_replay/heldout_replay.json` (the replay result, including the
matched probe comparison). Running either script against this clone will stop
on the completeness guard in `check_eval_half.py`. Both scripts are listed under
the measurement path in [REPRODUCING.md](REPRODUCING.md), not the laptop path,
for this reason.

**The winner's-curse figures.** See below: no released script computes them.

**`analyse_search_space.py`** reads the unreleased raw stage CSVs and exits with
a message if they are absent. `verify_section4.py` computes the same Section IV
numbers from the caches.

Everything else in the Quickstart runs from `results/` as committed.

---

## Which baseline is which

Before anything else. `libra_select.bl_race_only` is **not** the full-space
halving baseline the paper reports. It appears in
`results/per_seed_regrets.csv` as `race_only_legacy` and races a random
power-of-two subset of the space, 64 of 103 at a 960-query budget. The paper's
baseline is `full_space_halving.bl_successive_halving`, which races every
configuration and reallocates the remaining budget across the remaining rounds.

At 960 queries on Llama-3.1 the two give 5.58 and 0.23 points. Confusing them
changes every comparison in Section V. The 5.58 is what a random field of 64
predicts, since that target's optimum is a single component 14.4 points clear
of the best composed cell: (1 - 64/103) × 14.4 = 5.45.

`reviewer_analyses.py` wires all five baselines correctly, and
`heldout_replay.py` checks each of them against `results/per_seed_regrets.csv`
before it will produce a number.

---

## Robustness checks

**The instruction bootstrap** (`bootstrap_search.py`). The headline comparison
averages over seeds with the probe half fixed, which answers "would this run
again" rather than "would this hold on other instructions". The bootstrap
resamples the 180 probe instructions with replacement and re-selects the
configuration inside every resample. A 180-draw resample retains about 114
distinct source instructions, and multiplicities are preserved: the helper
indexes the outcome matrix by the draw vector and relabels every column, so
duplicate draws remain separate columns.

At 120 queries the advantage is +23.3 points, percentile interval
[18.3, 28.6], positive in 200 of 200 resamples. **At 240 queries the interval
contains zero** ([-1.8, +3.7]). The seed-level paired test at 240 is
significant; the instruction-level one is not. The two are reported separately
because they resample different units.

**The held-out replay** (`heldout_replay.py`). The whole procedure and all five
baselines replayed on the 340 evaluation instructions for the four open-weight
targets, scored against the held-out optimum. LIBRA-S's regret stays within
0.84 points of its probe-grid figure at every budget, and the advantage over
full-space halving is significant at five of six budgets. The exception is 360
queries, where the difference is -0.15 points on held-out and -0.06 on the
matched probe grid; an exception that reproduces on an independent instruction
set is worth reporting as one.

**The allocation sweep** (`sweep_rho_nr.py`). The screening fraction and the
race's starting replication were fixed a priori rather than tuned. The script
gates on reproducing the published Table 6 anchors (6.427 at 240 queries, 2.286
at 960) before sweeping anything. **Some setting beats the fixed values at four
of seven budgets**; at 960 a screening fraction of 0.5 reaches 1.13 against
2.29 while spending fewer queries (828 against 960). We report this rather than
adopting the better settings: re-tuning on the same five targets we evaluate on
would turn the reported regret into a tuned quantity.

**Winner's curse.** Selecting a configuration on a set of instructions and
scoring it on those same instructions inflates its measured attack success. On
a bootstrap of the probe instructions that re-picks the optimum inside each
resample, the selected configuration averages 65.5% in-bag and 63.5%
out-of-bag, an inflation of about 2 points.

The **transfer gap is insensitive to this**, because the bias inflates the
target's own optimum and the peer configurations alike: 33.1 points scoring
in-bag against 33.2 out-of-bag. Both sit slightly above the full-grid 32.6 of
the main text, which fixes each target's optimum on the whole grid. These are
different estimators rather than corrections of one another, and the spread is
well inside the interval, [27.6, 38.8] out-of-bag.

**No released script computes these two figures.** `bootstrap_search.py`
resamples instructions for the search comparison only, and
`reviewer_analyses.py`'s transfer bootstrap holds each target's optimum fixed
outside the resample loop. The numbers above were computed ad hoc from
`results/replay_cache.parquet` and are stated here so that nobody looks for
them in the code.

**The held-out grid** (`analyse_eval_half.py`). Selecting on the probe half and
measuring on the evaluation half costs 0.00, 0.00, 1.18 and 0.00 points across
the four open-weight targets, mean 0.29. The probe optimum remains optimal on
three of the four, and per-configuration rankings correlate at Spearman 0.885
to 0.972. Thirteen configuration-target grids were measured twice during the
run; all thirteen agree cell-by-cell on `success_strict`.

---

## The screening design

The search screens a fixed subset of 22 composed cells. Which 22 is a design
choice, and three scripts bear on it.

`coverage_design.py` is the generator. It maximises distinct (framing,
template) pairs, then (framing, order) pairs, then evenness of template and
order counts:

```bash
python coverage_design.py     # reaches 22/44 framing-template, 22/22 framing-order
```

The subset actually used, `libra_select.SCREEN_DESIGN`, attains 22/44
framing-template but **18/22** framing-order pairs: four framings receive both
of their cells at the same composition order. **The generator released here
does not reproduce it.** We report the subset we evaluated rather than
substituting one we did not run, and make no claim about which was constructed
first: a different generator output establishes non-reproduction, not
chronology.

The exhaustive grid of Section IV contains both orders for every
framing-template pair, so the order result there is unaffected. Section V-F
compares the archived subset against particular alternatives; it does not
isolate the causal cost of the four missing pairs, since an alternative design
changes which cells are sampled as well as how many pairs it covers.

`table6_paired.py` sweeps the screening size against budget with Holm-adjusted
paired tests over seed blocks. Every design's exact cell list and SHA-256 are
written to `results/table6_designs.json` before anything uses them.

### evolve_design.py is exploratory

`evolve_design.py` and `results/evolved_design/` are **not evidence for
anything in the paper.** The script asks whether a design chosen by fitness
rather than by a structural criterion would do better. Fitness is regret plus a
coverage penalty, `lam` points per missing pair, so it is not a pure regret
objective. It evolves on four targets and measures on the fifth, rotating
across all five.

```bash
python3 evolve_design.py --lam 0.25 --final
```

30 training seeds, 200 evaluation seeds, and the script's own
`default_rng(seed)` rule, which is **not** the `10000 * budget + seed` rule the
paper's results use. The outcome is mixed and supports no conclusion either
way: at 240 queries the evolved designs average 6.16 points against the
production design's 6.44 and win on three of five held-out targets, while at
960 they average 2.39 against 2.35 and win on one of five. Those are the
zero-based cohort's figures, not the paper's 6.43 and 2.29, which come from the
budget-offset cohort; the two must not be mixed. It is a single run per fold
with no paired tests.

An earlier version of this folder, produced in September against the selector
as it stood before the budget-cap and stable-tie fixes, reported the opposite
sign at 240. It is not in this repository. Do not cite either version as
evidence that greater pair coverage cannot help: the evolved designs reach 21
or 22 of the 22 framing-order pairs where the production design reaches 18, and
at the lower budget they do better, which is consistent with Table 6.

`results/evolved_design/final_design.json` is a *different* 22-cell list
(SHA-256 `da7b9f8a7b97`, against the production design's `f75b5ca49866`) and is
**not** used anywhere in the paper.

---

## Reproduction is not bit-exact

An independent replay on a different machine, with the same numpy and pandas
versions, matched all 30,000 main-method runs but differed on 5 of the 35,000
Table 6 selections, moving four aggregate cells by at most 0.035 points and
changing no significance decision. All five reproduce on the machine that wrote
the CSV. The mechanism is consistent with floating-point accumulation order but
has not been established.

Running with fewer than 200 seeds will differ by a few tenths of a point and
may show small non-monotonicities in the budget curve; these are seed noise
rather than a property of the procedure.

---

## Notes for anyone re-running the measurement

**Report the generation interface.** Raw completion and native chat templating
are different input conditions. In the archived Llama-2-7B-Chat diagnostic, the
technical-manual framing succeeds on 175 of 180 probe instructions (97.2%) as a
raw completion, in each of two independent runs, and on none of the same 180
submitted through the model's chat template. `models/model_wrapper.py` applies each
model's native template by default; `--no-chat-template` enables
raw-completion mode.

**Two targets need a different environment.** Ministral-3-8B requires
`mistral-common` and a recent `transformers` for `MistralCommonBackend`. Pass
`CONDA_ENV=...` to the SLURM scripts to override.

**Two AdvBench instructions are filtered by the API provider** before
generation. Both fall in the probe half, so API-served targets are measured on
178 probe instructions and all 340 evaluation instructions.

**Only answer text is scored.** Explicitly delimited reasoning segments are
removed for Qwen3-14B. For GPT-5.1, the batch parser extracts answer text from
message output items and excludes other response items.

**SLURM scripts are site-specific.** Set `SLURM_ACCOUNT`, and optionally
`SLURM_MAIL`, `PROJECT_DIR` and `HF_HOME`. Partition names and module load
lines will need editing.

**Expect provider-side safety systems to fire.** We received a usage-policy
warning during this work; the affected traffic was 13 requests out of roughly
47,000, recorded in `results/provider_filter_rejections.json` with request
bodies and provider request IDs omitted.
