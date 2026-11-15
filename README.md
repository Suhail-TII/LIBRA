# LIBRA-S: Query-Efficient Search for Target-Specific Jailbreak Configurations

Suhail Al Marzouqi, Maha Kadadha, Martin Andreoni, Hadi Otrok, Rabeb Mizouni
Khalifa University, Abu Dhabi · Technology Innovation Institute, Abu Dhabi

**LIBRA** (Layered Instruction Bracketing with Reordered Assembly) is a harmful
instruction wrapped in a semantic framing and a structural template, in either
composition order. Eleven framings, four templates and two orders give 88
compositions; with the 15 standalone components, a space of 103 configurations.

**LIBRA-S** is the budgeted search over that space. It measures a fixed subset,
predicts the rest from per-component effects, and races a shortlist by
successive halving, using only black-box access and no auxiliary attacker model.

Because all 103 configurations were measured exhaustively on five targets, every
target's optimum is known and regret is computable. No two targets share an
optimum, and reusing a peer's forfeits 32.6 points of attack success on average.

<img src="figures/fig_regret.png" width="640" alt="Regret against query budget">

## Key features

- **An exhaustive reference.** 103 configurations × 180 probe instructions × 7
  models, 129,162 measured outcomes, so regret is measured rather than estimated.
- **A budgeted search.** Within 6.4 points of the optimum at 240 queries and 2.3
  at 960, against 18,540 for exhaustive measurement.
- **A held-out evaluation.** The search and all five baselines replayed on 340
  unseen instructions for the four open-weight targets.
- **Replay, not re-attack.** Reproducing the search results needs no GPU, no
  model weights and no API key.

## Installation

```bash
conda env create -f environment.yml && conda activate libra
# or:  pip install -r requirements.txt

python -c "import config, attacks, models, utils; print('ok')"
```

The import check confirms the clone is intact and that the analysis path pulls
in no torch dependency.

## Quickstart

Reproduces the paper's tables and figures from the released caches, on a laptop,
in minutes:

```bash
python verify_section4.py                                   # Tables 2-4
python libra_select.py --lomo --budgets 120,240,360,480,720,960 --seeds 200
python reviewer_analyses.py --boot 2000 --seeds 200         # Section V baselines
python table6_paired.py --seeds 200                         # Table 6
python bootstrap_search.py --boot 2000 --seeds 200          # Section V-C
python sweep_rho_nr.py --check                              # Section V-E
python make_figures.py --outdir figures                     # Figures 2, 3
python make_tables.py                                       # Tables 5, 6 + checks
```

Run `make_figures.py` and `make_tables.py` last; both read CSVs the earlier
scripts write. `make_tables.py` re-derives 99 reference values and exits
non-zero if any fails.

Results average 200 seeds. Fewer will differ by a few tenths of a point.

## Data

`results/replay_cache.parquet` is the measurement: one row per (model,
configuration, instruction) with a boolean outcome.

| model | config | instruction_id | success | kind |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | `plinius_format_hijack` | 42 | True | framing_only |
| Qwen3-14B | `libra_style\|json\|libra_then_utos` | 42 | False | composed |

`results/judge_cache.parquet` holds per-response classifier verdicts for all
128,414 scored responses. `PROMPTS.md` renders every framing, template and
composition order on an example instruction.

**Not released:** the raw generations, and the per-cell outcomes of the held-out
grid. Two sections therefore cannot be regenerated from this clone; see
[NOTES.md](NOTES.md#what-cannot-be-reproduced-here).

## Repository structure

- `attacks/`, `models/`, `analysis/` — the framing and template registries, the
  generation wrapper, and the two safety classifiers.
- `results/` — the released caches and every per-run file behind a table.
- `figures/`, `tables/` — generated output.
- `slurm/` — site-specific job scripts for the measurement path.

## Documentation

- [REPRODUCING.md](REPRODUCING.md) — both paths in full: the laptop replay, and
  the measurement from scratch with GPU and API access. Per-script inputs and
  outputs.
- [NOTES.md](NOTES.md) — what cannot be reproduced here, which baseline is
  which, the robustness checks as they came out, and the screening design's
  known non-reproduction.
- [DISCLOSURE.md](DISCLOSURE.md) — what is released, what is withheld, and why.
- [PROMPTS.md](PROMPTS.md) — the full configuration text.

## Citation

```bibtex
@article{almarzouqi2026libra,
  title   = {{LIBRA-S}: Query-Efficient Search for Target-Specific
             Jailbreak Configurations},
  author  = {Al Marzouqi, Suhail and Kadadha, Maha and Andreoni, Martin
             and Otrok, Hadi and Mizouni, Rabeb},
  journal = {IEEE Access},
  year    = {2026}
}
```

## Disclaimer

This repository characterizes an attack surface in order to inform defense. The
evaluated behaviors come from AdvBench, a public benchmark; model weights and
safety training are not modified. The configuration text is released and some
of it cleared 90% attack success under our evaluator. Reproducing the
measurement will trigger provider-side safety systems: use a dedicated account
and do not resubmit prompts that return content-filter errors.

## License

Code MIT, data CC BY 4.0. `third_party/l1b3rt4s/` is AGPL-3.0 and holds
framings derived from the L1B3RT4S corpus; see that directory's NOTICE for
provenance and measured overlap. AdvBench is redistributed under its original
terms.
