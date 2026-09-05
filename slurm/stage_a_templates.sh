#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# slurm/stage_a_templates.sh  —  Stage A-T: UTOS template screening
#
#   sbatch --array=0,1,5 slurm/stage_a_templates.sh   # open-weight
#   sbatch --array=0-5   slurm/stage_a_templates.sh   # all six (GPT run sync)
#
# 12 templates x 180 probe instructions = 2,160 generations per model --
# about a quarter of the framing screen, so a single segment is ample.
#
# Purpose: re-derive UTOS_TOP4 on probe data. The current top-4 was chosen on
# the old exp4 run, on the same instructions the paper reports on. This also
# fixes the Stage B grid size: 11 framings x k templates x 2 orders.
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
#SBATCH --job-name=libra_stage_at
#SBATCH --output=logs/stage_at_%A_%a.out
#SBATCH --error=logs/stage_at_%A_%a.err
#SBATCH --mail-type=END,FAIL
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

OUTPUT_DIR=${OUTPUT_DIR:-results/stage_a_templates}
SPLIT=${SPLIT:-config/advbench_split_v1.json}

echo "======================================================================"
echo "Stage A-T  : UTOS template screening"
echo "Job        : $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Model      : $MODEL"
echo "Output     : $OUTPUT_DIR"
echo "Node       : $SLURMD_NODENAME"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

if [[ $SLURM_ARRAY_TASK_ID -ge 2 && $SLURM_ARRAY_TASK_ID -le 4 ]]; then
    sleep $(( (SLURM_ARRAY_TASK_ID - 2) * 120 ))
fi

FLAGS="--model $MODEL --split $SPLIT --output-dir $OUTPUT_DIR --resume"
[[ "$RAW" == "1" ]] && FLAGS="$FLAGS --no-chat-template"
[[ -n "$MAXTOK" ]] && FLAGS="$FLAGS --max-tokens $MAXTOK"
[[ -n "$TEMPLATES" ]] && FLAGS="$FLAGS --templates $TEMPLATES"
[[ -n "$EVAL_DEVICE" ]] && FLAGS="$FLAGS --evaluator-device $EVAL_DEVICE"

echo "Command: python screen_templates.py $FLAGS"
echo "----------------------------------------------------------------------"
python screen_templates.py $FLAGS
STATUS=$?
echo "----------------------------------------------------------------------"
echo "Exit status: $STATUS"
exit $STATUS