#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scrub_identifiers.sh — replace site-specific values with environment
# variables so the SLURM scripts are portable and carry no personal details.
#
# The scripts hardcode a cluster account, an institutional email, and an
# absolute project path. None of those are secrets, but they identify the
# author and are useless to anyone else, so they become overridable variables
# with neutral defaults.
#
# Run from the artifact directory. Prints a diff summary and does not commit.
#
#   bash scrub_identifiers.sh          # apply
#   bash scrub_identifiers.sh --check  # report only, change nothing
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CHECK=0
[[ "${1:-}" == "--check" ]] && CHECK=1

# what to look for, and what it becomes
ACCOUNT="${SLURM_ACCOUNT}"
EMAIL="your.email@institution.edu"
PROJECT="${PROJECT_DIR}"
USERHOME="<path>"

echo "scanning for site-specific identifiers..."
HITS=$(grep -rl "$ACCOUNT\|$EMAIL\|$USERHOME" \
        --include='*.sh' --include='*.py' --include='*.yml' \
        --include='*.yaml' --include='*.md' . 2>/dev/null || true)

if [[ -z "$HITS" ]]; then
    echo "  none found."
    exit 0
fi

echo "found in:"
echo "$HITS" | sed 's/^/  /'

if [[ $CHECK -eq 1 ]]; then
    echo ""
    echo "--check: showing occurrences, nothing modified"
    grep -rn "$ACCOUNT\|$EMAIL\|$USERHOME" \
        --include='*.sh' --include='*.py' --include='*.yml' \
        --include='*.yaml' --include='*.md' . 2>/dev/null | head -40
    exit 0
fi

echo ""
echo "rewriting..."
while IFS= read -r f; do
    [[ -z "$f" ]] && continue

    # SBATCH directives become overridable. The default is a placeholder that
    # will fail loudly rather than silently submitting to someone else's queue.
    sed -i "s|#SBATCH --account=$ACCOUNT|#SBATCH --account=\${SLURM_ACCOUNT:-CHANGE_ME}|g" "$f"
    sed -i "s|#SBATCH --mail-user=$EMAIL|#SBATCH --mail-user=\${SLURM_MAIL:-}|g" "$f"

    # Paths become variables with a repo-relative default.
    sed -i "s|PROJECT_DIR=$PROJECT|PROJECT_DIR=\${PROJECT_DIR:-\$PWD}|g" "$f"
    sed -i "s|$PROJECT/hf_cache|\${HF_HOME:-\$PWD/hf_cache}|g" "$f"
    sed -i "s|$PROJECT/hf_datasets|\${HF_DATASETS_CACHE:-\$PWD/hf_datasets}|g" "$f"
    sed -i "s|$PROJECT/LIBRA|\${PROJECT_DIR:-\$PWD}|g" "$f"
    sed -i "s|$USERHOME[^ \"']*|<path>|g" "$f"

    # Anything left over.
    sed -i "s|$EMAIL|your.email@institution.edu|g" "$f"
    sed -i "s|$PROJECT|\${PROJECT_DIR}|g" "$f"
    sed -i "s|$ACCOUNT|\${SLURM_ACCOUNT}|g" "$f"

    echo "  ~ $f"
done <<< "$HITS"

echo ""
echo "remaining occurrences (should be none):"
grep -rn "$ACCOUNT\|$EMAIL\|$USERHOME" \
    --include='*.sh' --include='*.py' --include='*.yml' \
    --include='*.yaml' --include='*.md' . 2>/dev/null || echo "  clean"

echo ""
echo "Next: check one script reads sensibly, then amend the commit."
echo "  head -40 slurm/stage_a.sh"
