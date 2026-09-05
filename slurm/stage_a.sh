#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# slurm/stage_a.sh  —  LIBRA Stage A: standalone framing screening
#
# Submit:
#   sbatch --array=0-6 slurm/stage_a.sh                    # all seven models
#   sbatch --array=1   slurm/stage_a.sh                    # Llama-3.1 only
#   sbatch --array=1 --export=ALL,SMOKE=1 slurm/stage_a.sh # 5-instruction test
#
# Each array task screens ALL framing candidates against ONE model on the
# PROBE half of config/advbench_split_v1.json. Tasks are independent -- no
# dependencies, no chaining, no shared state.
#
# Runtime is bounded: 25 framings x 180 instructions = 4,500 generations plus
# judge passes. Open-weight models should finish well inside 12h; the GPT
# tasks are API-bound and mostly idle, so they still need a GPU because the
# judges (Llama Guard 8B + HarmBench 13B) run locally.
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
#SBATCH --job-name=libra_stage_a
#SBATCH --output=logs/stage_a_%A_%a.out
#SBATCH --error=logs/stage_a_%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=${SLURM_MAIL:-}

set -o pipefail

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load intel_h200_gpu
module load cuda/12.4
module load miniconda/3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cca-jail-g4

echo "Python    : $(which python)"
echo "Conda env : $CONDA_DEFAULT_ENV"

# ── Storage ──────────────────────────────────────────────────────────────────
PROJECT_DIR=${PROJECT_DIR:-$PWD}
export HF_HOME=$PROJECT_DIR/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$PROJECT_DIR/hf_datasets
mkdir -p $HF_HOME $HF_DATASETS_CACHE logs

# API keys
if [ -f "$SLURM_SUBMIT_DIR/.env" ]; then
    set -a; source "$SLURM_SUBMIT_DIR/.env"; set +a
else
    echo "WARNING: .env not found at $SLURM_SUBMIT_DIR/.env" >&2
fi

# Gated repos (Llama-Guard-3-8B, Llama-2, Llama-3.1) need auth even when
# cached, because transformers checks the Hub for config.json first.
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    hf auth login --token "$HUGGINGFACE_TOKEN" 2>/dev/null \
        && echo "HF auth: ok" || echo "HF auth: FAILED" >&2
fi

# Everything needed is already cached. Offline mode turns a silent 15GB
# re-download into an immediate error, which is what you want inside a
# long grid run.
export HF_HUB_OFFLINE=1

unset PYTORCH_CUDA_ALLOC_CONF
cd $SLURM_SUBMIT_DIR

# ── Model list (array index 0-6) ─────────────────────────────────────────────
MODEL_LIST=(
    "Llama-2-7B-Chat"
    "Llama-3.1-8B-Instruct"
    "gpt-5-mini"
    "gpt-5-nano"
    "gpt-5.1"
    "Qwen3-14B"
    "Gemma-4-12B-it"
    "Ministral-3-8B-Instruct"
)

MODEL=${MODEL_LIST[$SLURM_ARRAY_TASK_ID]}
if [ -z "$MODEL" ]; then
    echo "No model for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

OUTPUT_DIR=${OUTPUT_DIR:-results/stage_a}
SPLIT=${SPLIT:-config/advbench_split_v1.json}

echo "======================================================================"
echo "Stage A    : framing screening"
echo "Job        : $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Model      : $MODEL"
echo "Split      : $SPLIT (probe half only)"
echo "Output     : $OUTPUT_DIR"
echo "Node       : $SLURMD_NODENAME"
echo "GPU        : $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

# ── Stagger GPT tasks so three jobs don't hit the API at once ────────────────
if [[ $SLURM_ARRAY_TASK_ID -ge 2 && $SLURM_ARRAY_TASK_ID -le 4 ]]; then
    DELAY=$(( (SLURM_ARRAY_TASK_ID - 2) * 120 ))
    echo "API model — staggering start by ${DELAY}s"
    sleep $DELAY
fi

# ── Flags ────────────────────────────────────────────────────────────────────
FLAGS="--model $MODEL --split $SPLIT --output-dir $OUTPUT_DIR --resume"

# SMOKE=1 -> 5 instructions, separate output dir so it can't pollute the real
# checkpoint. A smoke checkpoint sharing a path with the real run would make
# the real run resume from 125 rows and skip them.
if [[ "$SMOKE" == "1" ]]; then
    FLAGS="--model $MODEL --split $SPLIT --output-dir results/stage_a_smoke --limit 5"
    echo "SMOKE TEST: 5 instructions, output -> results/stage_a_smoke, no resume"
fi

if [[ "$RAW" == "1" ]]; then
    FLAGS="$FLAGS --no-chat-template"
    echo "RAW COMPLETION MODE: no chat template (template-ablation arm)"
fi

[[ -n "$FRAMINGS" ]] && FLAGS="$FLAGS --framings $FRAMINGS"
[[ -n "$EVAL_DEVICE" ]] && FLAGS="$FLAGS --evaluator-device $EVAL_DEVICE"

[[ -n "$MAXTOK" ]] && FLAGS="$FLAGS --max-tokens $MAXTOK"

echo "Command: python screen_framings.py $FLAGS"
echo "----------------------------------------------------------------------"

python screen_framings.py $FLAGS

STATUS=$?
echo "----------------------------------------------------------------------"
echo "Exit status: $STATUS"

if [ $STATUS -eq 0 ] && [ "$SMOKE" != "1" ]; then
    echo "Profile written to $OUTPUT_DIR/stage_a_profile_*.csv"
fi

exit $STATUS