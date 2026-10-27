#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# make_artifact.sh — assemble a clean, public artifact from the working repo.
#
# The working tree carries six generations of experiments, several abandoned
# analyses and a lot of one-off diagnostics. Publishing it as-is would make
# reproduction harder rather than easier, so this copies only what a reader
# needs and leaves the rest behind.
#
# WHAT IS PUBLISHED
#   results/replay_cache.parquet is the key artifact. It holds one row per
#   (model, configuration, instruction) with a boolean outcome, and reproduces
#   every number in the paper. It contains NO generated text, so releasing it
#   discloses no harmful content.
#
# WHAT IS NOT PUBLISHED
#   The raw generations (results/stage_*/*.csv, several hundred MB) hold model
#   completions to harmful instructions. Those stay on the cluster.
#
# ON FAILURE
#   A missing required file is fatal. It used to print a warning to stderr and
#   carry on, which is not an error under `set -e`, so the build exited 0 with
#   an incomplete artifact and the warning scrolled past a hundred "+ path"
#   lines. That is how a copy-list gap survives unnoticed for weeks.
#
#   Usage:  bash make_artifact.sh [target_dir]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${1:-$SRC/../libra-artifact}"

echo "source : $SRC"
echo "target : $DST"
mkdir -p "$DST"
DST="$(cd "$DST" && pwd)"

[[ "$DST" != "/" && "$DST" != "$HOME" && "$DST" != "$SRC" ]] \
    || { echo "refusing to build into $DST" >&2; exit 1; }

MISSING=()

copy() {  # copy() <relative-path> [required]
    local rel="$1" req="${2:-}"
    if [[ -e "$SRC/$rel" ]]; then
        mkdir -p "$DST/$(dirname "$rel")"
        if [[ -d "$SRC/$rel" ]]; then
            # cp -r of a directory onto an existing directory nests it
            # ($DST/figures/figures), so clear the target first. $rel is
            # always a literal from the list below, never user input.
            rm -rf "${DST:?}/$rel"
            cp -r "$SRC/$rel" "$DST/$rel"
        else
            cp "$SRC/$rel" "$DST/$rel"
        fi
        echo "  + $rel"
    elif [[ -n "$req" ]]; then
        echo "  ! MISSING (required): $rel" >&2
        MISSING+=("$rel")
    else
        echo "  - absent, not required: $rel"
    fi
}

echo ""
echo "── design space ────────────────────────────────────────────"
copy attacks/__init__.py            req
copy attacks/framing_registry.py    req   # the 43-candidate pool and dedupe
copy attacks/libra_framings.py      req   # 8 exp5 framings, 12 UTOS templates
copy attacks/obfuscation_v2.py      req   # struct_* framings, imported above
copy attacks/l1b3rt4s_attacks.py    req   # plinius_* framings, imported above

echo ""
echo "── evaluation and generation ───────────────────────────────"
copy analysis/__init__.py       req
copy analysis/metrics.py        req   # Llama Guard + HarmBench + strict/lenient
copy models/__init__.py         req
copy models/model_wrapper.py    req   # chat templating, per-family branches
copy models/loader.py           req   # load_model_from_config
copy utils/__init__.py          req
copy utils/checkpoint.py        req

echo ""
echo "── configuration ───────────────────────────────────────────"
copy config/__init__.py
copy config/models.yaml               req
copy config/advbench_split_v1.json    req   # the frozen 180/340 split
copy build_advbench_split.py          req   # how that split was built

echo ""
echo "── measurement pipeline ────────────────────────────────────"
copy screen_framings.py     req   # Stage A
copy screen_templates.py    req   # Stage A-T
copy run_stage_b.py         req   # Stage B, and the 11-framing definition
copy batch_stage_a.py       req   # API path for all three stages
copy decode_and_rejudge.py  req   # encoded-output correction
copy utils/dataset_loader.py  req

echo ""
echo "── analysis (reproduces the paper) ─────────────────────────"
copy analyse_search_space.py  req   # Section IV
copy verify_section4.py       req   # Section IV, every number
copy libra_select.py          req   # Section V, the search procedure
copy reviewer_analyses.py     req   # Section V, baselines and diagnostics
copy coverage_design.py       req   # the generator Section V-C says we release
copy table6_paired.py         req   # Table VI: screening sweep, paired tests
copy run_eval_half.py         req   # probe/eval check: selection and analysis
copy run_eval_half_gen.py     req   # probe/eval check: generation
copy make_figures.py          req   # the two paper figures
copy make_tables.py           req   # Tables V and VI, and the number check
copy check_manuscript_numbers.py req  # the manuscript's own numbers vs results/
copy verify_paper_numbers.py
# only needed if reviewer_analyses.py imports it rather than defining
# bl_successive_halving itself, so not marked required
copy full_space_halving.py
# exploratory, not used in the paper, but the README describes it and a
# rebuild must not silently drop what the README points a reader at
copy evolve_design.py

echo ""
echo "── repository hygiene ──────────────────────────────────────"
copy make_artifact.sh              # how this artifact was assembled
copy audit_artifact_coverage.sh    # what it does and does not copy

echo ""
echo "── cluster scripts ─────────────────────────────────────────"
copy slurm/stage_a.sh
copy slurm/stage_a_templates.sh
copy slurm/stage_b.sh
copy slurm/judge_batch.sh
copy slurm/judge_cells.sh
copy slurm/decode_rejudge.sh
copy slurm/submit_stage_a.sh
copy slurm/eval_half.sh

echo ""
echo "── data ────────────────────────────────────────────────────"
copy results/replay_cache.parquet  req
copy results/judge_cache.parquet   req
copy results/eval_half_cache.parquet
copy results/eval_half/selection.json
copy results/provider_filter_rejections.json
copy results/search_space.csv
copy results/lomo_results.csv
copy results/reviewer_analyses.csv

# produced by the revised sweeps. Figure 3's bands, Table V and Table VI are
# regenerated from these, and make_tables.py checks the paper against them.
# results/regret_seeds.csv is deliberately NOT published: it is the superseded
# per-seed export and a reader who found it would reproduce the old numbers.
copy results/per_seed_regrets.csv  req
copy results/table6_per_seed.csv   req
copy results/table6_paired.csv     req
copy results/table6_designs.json   req
copy results/evolved_design

echo ""
echo "── generated outputs ───────────────────────────────────────"
# published so a reader can diff their regeneration against ours rather than
# only eyeballing it. Regenerate before building, or these go stale.
copy figures
copy tables

echo ""
echo "── environment ─────────────────────────────────────────────"
copy environment.yml    req
copy requirements.txt   req   # carries pyarrow, without which the parquet
                              # caches cannot be opened

echo ""
echo "── documentation ───────────────────────────────────────────"
copy README.md      req
copy LICENSE        req
copy LICENSE-DATA
copy PROMPTS.md     req   # the full framing and template text
copy REPRODUCIBILITY.json  req   # the generation and evaluation manifest

# ── generated files ──────────────────────────────────────────────
mkdir -p "$DST/results"
cat > "$DST/.gitignore" <<'EOF'
__pycache__/
*.pyc
.env
.claude/
logs/
hf_cache/
hf_datasets/

# raw generations: not released, see README
results/stage_a/
results/stage_a_raw/
results/stage_a_templates/
results/stage_a_batch/
results/stage_a_decoded/
results/stage_b/
results/stage_b_decoded/
results/stage_a_cap2048/
results/*_smoke/
EOF

echo ""
echo "── cleaning ────────────────────────────────────────────────"
find "$DST" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$DST" -name '*.pyc' -delete 2>/dev/null || true

echo ""
echo "size:"
du -sh "$DST"
du -sh "$DST"/* 2>/dev/null | sort -h | tail -5

if (( ${#MISSING[@]} )); then
    echo ""
    echo "FAILED: ${#MISSING[@]} required file(s) absent from the source tree:"
    printf '  %s\n' "${MISSING[@]}"
    echo ""
    echo "$DST is incomplete. Do not publish it, and do not rsync --delete"
    echo "over an existing artifact with it."
    exit 1
fi

echo ""
echo "Next:"
echo "  1. bash audit_artifact_coverage.sh --src $SRC $DST"
echo "       ORPHAN must be empty before rsync --delete is safe"
echo "  2. cd $DST && python verify_section4.py    # Section IV reproduces"
echo "  3. cd $DST && python make_tables.py        # the paper's numbers agree"
echo "  4. git add -A && git commit"