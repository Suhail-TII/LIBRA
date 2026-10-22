#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# slurm/eval_half.sh — generate the selected configurations on the held-out
# evaluation half.
#
#   sbatch --array=0,1,3 slurm/eval_half.sh                 # open-weight
#   sbatch --array=4 --export=ALL,CONDA_ENV=cca-jail-g4 slurm/eval_half.sh
#
# 10 configurations x 340 instructions = 3,400 generations per target, roughly
# a fifth of a Stage B run, so one segment is ample.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --account=kuex0037
#SBATCH --qos=normal
#SBATCH --job-name=libra_eval_half
#SBATCH --output=logs/eval_half_%A_%a.out
#SBATCH --error=logs/eval_half_%A_%a.err

set -o pipefail

module purge
module load intel_h200_gpu
module load cuda/12.4
module load miniconda/3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-cca-jail-g4}
echo "Conda env : $CONDA_DEFAULT_ENV"

PROJECT_DIR=${PROJECT_DIR:-/dpc/kuex0037}
export HF_HOME=${HF_HOME:-$PROJECT_DIR/hf_cache}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$PROJECT_DIR/hf_datasets}
mkdir -p logs
export HF_HUB_OFFLINE=1
cd $SLURM_SUBMIT_DIR

MODEL_LIST=(
    "Llama-2-7B-Chat"
    "Llama-3.1-8B-Instruct"
    "gpt-5.1"
    "Qwen3-14B"
    "Ministral-3-8B-Instruct"
)
MODEL=${MODEL:-${MODEL_LIST[$SLURM_ARRAY_TASK_ID]}}
[ -z "$MODEL" ] && { echo "No model for index $SLURM_ARRAY_TASK_ID" >&2; exit 1; }

echo "======================================================================"
echo "Eval half  : selected configurations on the 340 held-out instructions"
echo "Job        : $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Model      : $MODEL"
echo "Node       : $SLURMD_NODENAME"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

FLAGS="--model $MODEL --resume"
[[ -n "$MAXTOK" ]] && FLAGS="$FLAGS --max-tokens $MAXTOK"
[[ -n "$LIMIT" ]] && FLAGS="$FLAGS --limit $LIMIT"

echo "Command: python run_eval_half_gen.py $FLAGS"
echo "----------------------------------------------------------------------"
python run_eval_half_gen.py $FLAGS
STATUS=$?
echo "----------------------------------------------------------------------"
echo "Exit status: $STATUS"
exit $STATUS
