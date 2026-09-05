#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# slurm/submit_stage_a.sh  —  submit Stage A as chained segments
#
# Each segment is a full --array=0-5 job with a 24h walltime. Segment N+1 is
# submitted with --dependency=afterany:N, so it starts whether segment N
# succeeded, failed, or hit the walltime.
#
# Because screen_framings.py runs with --resume and checkpoints on a composite
# "{framing}_{instruction_id}" key, a later segment skips every cell already
# done and continues from where the previous one stopped. A segment that finds
# everything complete exits in a couple of minutes.
#
# afterany, not afterok: the whole point is to continue after a TIMEOUT, and a
# timed-out job does NOT satisfy afterok. The old submit_jobs.sh used afterany
# too, but combined it with --resume in a way that hid genuine failures --
# check the logs of every segment, not just the last.
#
# USAGE
#   bash slurm/submit_stage_a.sh                     # 6 models, 3 segments
#   bash slurm/submit_stage_a.sh --segments 5
#   bash slurm/submit_stage_a.sh --array 0,1,5       # open-weight only
#   bash slurm/submit_stage_a.sh --dry-run
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SEGMENTS=3
ARRAY="0-5"
DRY_RUN=0
SCRIPT="slurm/stage_a.sh"

while [[ $# -gt 0 ]]; do
    case $1 in
        --segments) SEGMENTS="$2"; shift 2 ;;
        --array)    ARRAY="$2";    shift 2 ;;
        --script)   SCRIPT="$2";   shift 2 ;;
        --dry-run)  DRY_RUN=1;     shift ;;
        *) echo "unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p logs

echo "======================================================================"
echo "Stage A submission"
echo "  script   : $SCRIPT"
echo "  array    : $ARRAY"
echo "  segments : $SEGMENTS  (24h each, chained with afterany)"
echo "  coverage : $((SEGMENTS * 24))h total per model"
echo "  dry run  : $DRY_RUN"
echo "======================================================================"

PREV=""
for ((s = 1; s <= SEGMENTS; s++)); do
    DEP=""
    [[ -n "$PREV" ]] && DEP="--dependency=afterany:$PREV"

    CMD="sbatch --parsable --array=$ARRAY $DEP $SCRIPT"
    echo ""
    echo "segment $s/$SEGMENTS"
    echo "  $CMD"

    if [[ $DRY_RUN -eq 1 ]]; then
        PREV="DRYRUN$s"
        continue
    fi

    JID=$($CMD)
    echo "  -> job $JID$([[ -n "$PREV" ]] && echo " (waits for $PREV)")"
    PREV="$JID"
done

echo ""
echo "======================================================================"
echo "monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/stage_a_*_1.out"
echo ""
echo "progress (rows per model in the checkpoints):"
echo "  wc -l results/stage_a/checkpoints/exp0_*_checkpoint.csv"
echo ""
echo "expected per model: 43 framings x 180 instructions = 7740 rows + header"
echo ""
echo "IMPORTANT: check every segment's log, not just the last. afterany means"
echo "segment N+1 starts even if segment N crashed, and --resume will happily"
echo "carry on from a partial checkpoint without surfacing the earlier error."
echo "======================================================================"