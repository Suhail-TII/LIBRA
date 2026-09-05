#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# slurm/stage_b.sh  —  Stage B: the composition grid
#
#   sbatch --array=0,1,5 slurm/stage_b.sh     # open-weight
#   sbatch --array=1     slurm/stage_b.sh     # Llama-3.1 only
#   sbatch --array=1 --export=ALL,SMOKE=1 slurm/stage_b.sh
#
# 88 cells x 180 probe instructions = 15,840 generations per model.
# Qwen3-14B runs ~8 rows/min -> ~33h, so it needs more than one segment.
# Chain with slurm/submit_stage_b.sh rather than submitting this directly.
#
# GPT models go through batch_stage_a.py --axis cells instead; running them
# here would be 15,840 sequential API calls at full price.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --account=${SLURM_ACCOUNT:-CHANGE_ME}
#SBATCH --qos=normal
#SBATCH --job-name=libra_stage_b
#SBATCH --output=logs/stage_b_%A_%a.out
#SBATCH --error=logs/stage_b_%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=${SLURM_MAIL:-}

set -o pipefail

module purge
module load intel_h200_gpu
module load cuda/12.4
module load miniconda/3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cca-jail-g4

PROJECT_DIR=${PROJECT_DIR:-$PWD}
export HF_HOME=$PROJECT_DIR/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$PROJECT_DIR/hf_datasets
mkdir -p $HF_HOME $HF_DATASETS_CACHE logs

if [ -f "$SLURM_SUBMIT_DIR/.env" ]; then
    set -a; source "$SLURM_SUBMIT_DIR/.env"; set +a
fi
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    hf auth login --token "$HUGGINGFACE_TOKEN" 2>/dev/null && echo "HF auth: ok"
fi
export HF_HUB_OFFLINE=1
unset PYTORCH_CUDA_ALLOC_CONF
cd $SLURM_SUBMIT_DIR

MODEL_LIST=(
    "Llama-2-7B-Chat"
    "Llama-3.1-8B-Instruct"
    "gpt-5-mini"
    "gpt-5-nano"
    "gpt-5.1"
    "Qwen3-14B"
    "Ministral-3-8B-Instruct"
)
MODEL=${MODEL_LIST[$SLURM_ARRAY_TASK_ID]}
[ -z "$MODEL" ] && { echo "No model for index $SLURM_ARRAY_TASK_ID" >&2; exit 1; }

OUTPUT_DIR=${OUTPUT_DIR:-results/stage_b}
SPLIT=${SPLIT:-config/advbench_split_v1.json}

echo "======================================================================"
echo "Stage B    : composition grid (88 cells)"
echo "Job        : $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Model      : $MODEL"
echo "Output     : $OUTPUT_DIR"
echo "Node       : $SLURMD_NODENAME"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

FLAGS="--model $MODEL --split $SPLIT --output-dir $OUTPUT_DIR --resume"

if [[ "$SMOKE" == "1" ]]; then
    FLAGS="--model $MODEL --split $SPLIT --output-dir results/stage_b_smoke --limit 3"
    echo "SMOKE TEST: 3 instructions -> results/stage_b_smoke, no resume"
fi

[[ "$RAW" == "1" ]]      && FLAGS="$FLAGS --no-chat-template"
[[ -n "$MAXTOK" ]]       && FLAGS="$FLAGS --max-tokens $MAXTOK"
[[ -n "$FRAMINGS" ]]     && FLAGS="$FLAGS --framings $FRAMINGS"
[[ -n "$TEMPLATES" ]]    && FLAGS="$FLAGS --templates $TEMPLATES"
[[ -n "$EVAL_DEVICE" ]]  && FLAGS="$FLAGS --evaluator-device $EVAL_DEVICE"

echo "Command: python run_stage_b.py $FLAGS"
echo "----------------------------------------------------------------------"
python run_stage_b.py $FLAGS
STATUS=$?
echo "----------------------------------------------------------------------"
echo "Exit status: $STATUS"
exit $STATUS