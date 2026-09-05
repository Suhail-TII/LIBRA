#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# slurm/decode_rejudge.sh  —  decode encoded responses and re-judge them
#
#   sbatch slurm/decode_rejudge.sh
#   sbatch --export=ALL,FRAMINGS=plinius_l33tspeak slurm/decode_rejudge.sh
#   sbatch --export=ALL,DETECT_ONLY=1 slurm/decode_rejudge.sh
#
# Both safety classifiers mislabel leetspeak: sampled plinius_l33tspeak
# responses are REFUSALS written in leet, yet Llama Guard flags 68.6% of them
# on gpt-5-mini. That framing is c* for all three GPT models, so the ground
# truth the selection algorithm is measured against is contaminated by judge
# failure rather than by attack success.
#
# This decodes the leet and re-judges, writing raw and decoded labels side by
# side. If decoded strict ASR collapses, the framing is an artifact and c*
# must be recomputed without it.
#
# Runtime: ~8,400 encoded responses at ~8/s plus model loads -> well under 1h.
# Offline only; no API traffic, safe to run under the OpenAI warning.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --account=${SLURM_ACCOUNT:-CHANGE_ME}
#SBATCH --qos=normal
#SBATCH --job-name=libra_decode
#SBATCH --output=logs/decode_%j.out
#SBATCH --error=logs/decode_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${SLURM_MAIL:-}

set -o pipefail

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
export HF_HUB_OFFLINE=1

cd $SLURM_SUBMIT_DIR
mkdir -p logs

INPUT_GLOB=${INPUT_GLOB:-'results/stage_b/stage_b_*.csv'}
OUT_DIR=${OUT_DIR:-results/stage_b_decoded}

echo "======================================================================"
echo "Decode + re-judge"
echo "Job    : $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Input  : $INPUT_GLOB"
echo "Output : $OUT_DIR"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

FLAGS=""
[[ -n "$FRAMINGS" ]]    && FLAGS="$FLAGS --framings $FRAMINGS"
[[ -n "$EVAL_DEVICE" ]] && FLAGS="$FLAGS --evaluator-device $EVAL_DEVICE"
[[ "$DETECT_ONLY" == "1" ]] && FLAGS="$FLAGS --detect-only"

echo "Command: python decode_and_rejudge.py --glob '$INPUT_GLOB' --out-dir $OUT_DIR $FLAGS"
echo "----------------------------------------------------------------------"

python decode_and_rejudge.py --glob "$INPUT_GLOB" --out-dir "$OUT_DIR" $FLAGS
STATUS=$?

echo "----------------------------------------------------------------------"
echo "Exit status: $STATUS"

if [ $STATUS -eq 0 ] && [ "$DETECT_ONLY" != "1" ]; then
    echo ""
    echo "Read the RAW vs DECODED table above."
    echo "  decoded strict collapses -> plinius_l33tspeak is a judge artifact;"
    echo "    rerun analyse_search_space.py with it excluded and c* will change"
    echo "    for the three GPT models, where it is currently the optimum."
    echo "  decoded strict holds     -> those cells are genuine attacks and the"
    echo "    search space stays as it is."
    echo "Either way the raw-vs-decoded gap is the Section VII finding."
fi

exit $STATUS