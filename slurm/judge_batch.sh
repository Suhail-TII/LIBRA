#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# slurm/judge_batch.sh  —  judge fetched GPT batch generations
#
#   sbatch slurm/judge_batch.sh                       # every gen_*.csv
#   sbatch --export=ALL,MODELS=gpt_5_1 slurm/judge_batch.sh
#
# Only the GPT models need this. screen_framings.py judges inline, so the
# open-weight results in results/stage_a/ already carry ensemble columns --
# re-judging them would just produce duplicate CSVs with new timestamps.
#
# Runtime: ~7 rows/s with Llama Guard 3-8B + HarmBench 13B loaded, so roughly
# 20 min per 7,740-row model, ~1h for all three including model loads.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --account=${SLURM_ACCOUNT:-CHANGE_ME}
#SBATCH --qos=normal
#SBATCH --job-name=libra_judge
#SBATCH --output=logs/judge_%j.out
#SBATCH --error=logs/judge_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${SLURM_MAIL:-}

module purge
module load intel_h200_gpu
module load cuda/12.4
module load miniconda/3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cca-jail

PROJECT_DIR=${PROJECT_DIR:-$PWD}
export HF_HOME=$PROJECT_DIR/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$PROJECT_DIR/hf_datasets
export HF_HUB_OFFLINE=1          # judges are cached; fail loudly if not

cd $SLURM_SUBMIT_DIR
mkdir -p logs

GEN_DIR=${GEN_DIR:-results/stage_a_batch}
OUT_DIR=${OUT_DIR:-results/stage_a}

# MODELS = space/comma separated suffixes (gpt_5_nano gpt_5_mini gpt_5_1).
# Unset -> judge every gen_*.csv present.
if [[ -n "$MODELS" ]]; then
    FILES=""
    for m in ${MODELS//,/ }; do FILES="$FILES $GEN_DIR/gen_${m}.csv"; done
else
    FILES=$(ls -1 $GEN_DIR/gen_*.csv 2>/dev/null)
fi

if [[ -z "$FILES" ]]; then
    echo "No generation files found in $GEN_DIR" >&2
    echo "Run: python batch_stage_a.py fetch --manifest <manifest>" >&2
    exit 1
fi

echo "======================================================================"
echo "Judging batch generations"
echo "Job    : $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Files  :"; for f in $FILES; do echo "         $f"; done
echo "Output : $OUT_DIR"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

FAILED=""
for f in $FILES; do
    if [[ ! -s "$f" ]]; then
        echo ">>> SKIP $f (missing or empty)"; FAILED="$FAILED $f"; continue
    fi
    echo ""
    echo ">>> $f  ($(python -c "import pandas as pd,sys;print(len(pd.read_csv(sys.argv[1])))" "$f") rows)"
    echo "----------------------------------------------------------------------"
    python batch_stage_a.py judge --input "$f" --output-dir "$OUT_DIR"
    rc=$?
    echo "--- exit $rc"
    [[ $rc -ne 0 ]] && FAILED="$FAILED $f"
done

echo ""
echo "======================================================================"
if [[ -n "$FAILED" ]]; then
    echo "FAILED:$FAILED"
    exit 1
fi
echo "All judged. Stage A CSVs in $OUT_DIR:"
ls -1 $OUT_DIR/stage_a_*.csv | grep -v profile
echo ""
echo "Six models expected: Llama-2, Llama-3.1, Qwen3-14B (judged inline by"
echo "screen_framings.py) plus gpt-5-nano, gpt-5-mini, gpt-5.1 (judged here)."
echo "======================================================================"