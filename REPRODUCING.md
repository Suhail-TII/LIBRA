# Reproducing

Two paths. The first regenerates the paper's results from the released caches
and needs a laptop. The second regenerates those caches from scratch and needs
a GPU and API access.

Read [NOTES.md](NOTES.md) first if you intend to cite a number from here. It
records what cannot be reproduced from this clone, which baseline is which, and
the checks that came out against us.

---

## Path 1: the paper, from the caches

Minutes, no GPU, no model weights, no API key.

```bash
conda env create -f environment.yml && conda activate libra
# or:  pip install -r requirements.txt

python -c "import config, attacks, models, utils; print('ok')"
```

```bash
python verify_section4.py                                   # Section IV
python libra_select.py --lomo --budgets 120,240,360,480,720,960 --seeds 200
python reviewer_analyses.py --boot 2000 --seeds 200         # Section V
python table6_paired.py --seeds 200                         # Table 6
python run_eval_half.py --analyse                           # probe/eval check
python bootstrap_search.py --boot 2000 --seeds 200          # Section V-C
python sweep_rho_nr.py --check                              # Section V-E anchor
python make_figures.py --outdir figures                     # Figures 2 and 3
python make_tables.py                                       # Tables 5 and 6
python check_manuscript_numbers.py --tex Chapters --results results
```

| script | produces |
|---|---|
| `verify_section4.py` | Tables 2-4; transfer cost, synergy, order, evaluator |
| `libra_select.py --lomo` | `results/lomo_results.csv`; the search under budget |
| `reviewer_analyses.py` | `results/per_seed_regrets.csv`, the baselines, the additive diagnostics of Section V-E, bootstrap CIs |
| `table6_paired.py` | `results/table6_*`; the screening-size sweep with Holm-adjusted paired tests |
| `run_eval_half.py --analyse` | probe vs held-out comparison for the ten selected configurations |
| `bootstrap_search.py` | `results/boot_search/boot_search.json`; the instruction bootstrap of Section V-C |
| `sweep_rho_nr.py` | `results/rho_nr_sweep/rho_nr_sweep.json`; the allocation sweep |
| `make_figures.py` | `figures/fig_heatmap.pdf` (Figure 2), `figures/fig_regret.pdf` (Figure 3) |
| `make_tables.py` | `tables/table5.tex`, `tables/table6.tex`, **and a 99-value reference check** |
| `check_manuscript_numbers.py` | reads the `.tex` and checks it against `results/` |

Run `make_figures.py` and `make_tables.py` last: both read CSVs the earlier
scripts write. `make_tables.py` exits non-zero if an embedded reference
comparison fails.

### What the two checkers do and do not do

`make_tables.py` compares CSV-derived values against reference constants
embedded in the script. **It does not parse the manuscript's TeX**, so it
catches drift in the data or the code, not an edit to a number in the paper.
`check_manuscript_numbers.py` is the complement: it reads the `.tex`,
recomputes each located entry from `results/`, and reports anything it could
not locate rather than skipping it.

### How replay works

`libra_select.py` replays the grid: the procedure may query only
(configuration, instruction) pairs that were actually measured, and receives the
recorded outcome. Replay needs no new target queries, and repeated seeds are
free. See [NOTES.md](NOTES.md#reproduction-is-not-bit-exact) on exactness and
seed counts.

---

## Path 2: the measurement, from scratch

Roughly 129,000 generations, each scored by two classifiers. Needs a GPU for the
open-weight targets and API access for the rest. On one H200 the open-weight
portion takes several days.

### 1. The instruction split

```bash
python build_advbench_split.py
```

Writes `config/advbench_split_v1.json`: 180 probe and 340 evaluation
instructions, partitioned by semantic cluster (all-mpnet-base-v2 embeddings,
agglomerative clustering with average linkage at cosine distance 0.35) so no
near-paraphrase spans both halves. **The frozen split used in the paper is
committed here** — rerunning this overwrites it and will not reproduce our
numbers exactly.

### 2. Screen the components (Stage A, Stage A-T)

```bash
export SLURM_ACCOUNT=your_account
export HF_HOME=/path/to/model/cache

sbatch --array=0,1,5 slurm/stage_a.sh              # 43 framings, open-weight
sbatch --array=0,1,5 slurm/stage_a_templates.sh    # 12 templates, open-weight

python batch_stage_a.py submit --model gpt-5.1 --axis framings
python batch_stage_a.py status --manifest results/stage_a_batch/manifest_gpt_5_1.json
python batch_stage_a.py fetch  --manifest results/stage_a_batch/manifest_gpt_5_1.json
sbatch slurm/judge_batch.sh
```

Stage A screens 43 framings and retains 11 (the union of each target's top
three). Stage A-T screens 12 templates and retains 4, ranked by mean per-model
rank. Both selections are made on the probe half only.

### 3. Measure the composition grid (Stage B)

```bash
bash slurm/submit_stage_a.sh --array 0,1,5 --segments 3 --script slurm/stage_b.sh
python batch_stage_a.py submit --model gpt-5.1 --axis cells
sbatch slurm/judge_cells.sh
```

88 cells (11 framings × 4 templates × 2 orders) × 180 instructions per target.

### 4. Correct encoded output

```bash
sbatch --export=ALL,INPUT_GLOB='results/stage_b/stage_b_*.csv' slurm/decode_rejudge.sh
sbatch --export=ALL,INPUT_GLOB='results/stage_a/stage_a_*.csv',OUT_DIR=results/stage_a_decoded slurm/decode_rejudge.sh
```

Two framings instruct the target to emit character-substituted output, which
both classifiers misread. This decodes and re-scores those responses. Skipping
it changes `c*` on all three API-served targets.

### 5. The held-out evaluation half

```bash
python run_eval_half.py --plan            # writes results/eval_half/selection.json
sbatch --array=0,1,3 slurm/eval_half.sh
sbatch --array=4 --export=ALL,CONDA_ENV=libra-mistral slurm/eval_half.sh
```

The ten-configuration plan above is what produced `eval_half_cache.parquet`.
The full held-out grid measures **all 103 configurations** on all 340
evaluation instructions for the four open-weight targets, 140,080
(configuration, instruction) pairs, which is what Sections IV-D and V-D report.
Check completeness before analysing, since a partial grid silently biases the
comparison:

```bash
python check_eval_half.py --root results/eval_half    # refuses unless complete
python inspect_eval_half.py --root results/eval_half --peek
python analyse_eval_half.py                           # writes heldout_103.csv
python heldout_replay.py --seeds 200 --matched         # Section V-D
```

The last two steps need the per-cell held-out outcomes, which are **not in this
repository** — they belong to this path, not Path 1. See
[NOTES.md](NOTES.md#what-cannot-be-reproduced-here).

### 6. Rebuild the caches

```bash
python libra_select.py --build-cache
```

---

## Repository layout

```
attacks/
  framing_registry.py     43 framing candidates, dedupe, loud/clean tagging
  libra_framings.py       12 UTOS templates, composition operators, STAGE_B_FRAMINGS
  obfuscation_v2.py       struct_* framings
  l1b3rt4s_attacks.py     plinius_* framings
analysis/
  metrics.py              Llama Guard 3-8B + HarmBench-13B, strict/lenient
models/
  model_wrapper.py        generation; chat templating, per-family branches
  loader.py               load_model_from_config
config/
  advbench_split_v1.json  the frozen probe/eval split (do not regenerate)
  models.yaml             target model registry
utils/
  checkpoint.py           resumable run state
  dataset_loader.py       AdvBench loading and the frozen split
results/
  replay_cache.parquet    103 configs x 180 instructions x 7 models, outcomes
  judge_cache.parquet     per-response classifier verdicts, 128,414 rows
  eval_half_cache.parquet the held-out half for the ten selected configurations
  per_seed_regrets.csv    every run behind Figure 3 and Table 5
  table6_*.{csv,json}     the screening-size sweep, its tests, its designs
  eval_half/              held-out grid summary (heldout_103.csv, selection.json)
  heldout_replay/         the held-out replay of Section V-D
  boot_search/            instruction-bootstrap intervals
  rho_nr_sweep/           allocation sweep, means and Holm-adjusted tests
  evolved_design/         exploratory, not used in the paper

measurement
  screen_framings.py      Stage A
  screen_templates.py     Stage A-T
  run_stage_b.py          Stage B
  batch_stage_a.py        API path for all three stages (--axis framings|templates|cells)
  decode_and_rejudge.py   encoded-output correction
  run_eval_half_gen.py    generation on the held-out evaluation half
  build_advbench_split.py the probe/eval partition

analysis
  verify_section4.py      Section IV, all numbers from the caches
  libra_select.py         Section V, the search procedure and SCREEN_DESIGN
  reviewer_analyses.py    Section V, baselines and the additive diagnostics
  full_space_halving.py   the full-space halving baseline (see NOTES.md)
  table6_paired.py        Table 6, the screening-size sweep with paired tests
  coverage_design.py      the screening-design generator
  evolve_design.py        exploratory, not used in the paper
  run_eval_half.py        probe/eval selection and comparison
  analyse_eval_half.py    the full 103-config held-out comparison
  heldout_replay.py       the selector and baselines on held-out instructions
  check_eval_half.py      completeness guard for the held-out grid
  inspect_eval_half.py    per-file diagnostic for the held-out output
  bootstrap_search.py     the instruction bootstrap of Section V-C
  sweep_rho_nr.py         the allocation sweep of Section V-E
  verify_paper_numbers.py cross-checks across the sweeps
  analyse_search_space.py Section IV from the raw CSVs (not released)

output
  make_figures.py         Figures 2 and 3
  make_tables.py          Tables 5 and 6, and the 99-value reference check
  check_manuscript_numbers.py  reads the .tex and checks it against results/
```
