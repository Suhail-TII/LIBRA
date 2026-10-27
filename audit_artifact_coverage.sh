#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# audit_artifact_coverage.sh — what the repo has that make_artifact.sh does
# not publish, and what a published artifact has that the script no longer
# produces.
#
# The copy list in make_artifact.sh is maintained by hand, so it drifts: a
# script written during the revision is committed to the repo and never added
# to the list. Nothing catches that, because a file absent from the list is
# simply never mentioned. The consequence only shows up later, when
# `rsync --delete` is used against an existing artifact and silently removes
# whatever the list forgot.
#
# This answers three questions:
#
#   GAP      tracked in git, never copied. Each one is either something that
#            should be published, or something that should be stated as
#            deliberately withheld. Neither is the current state, which is
#            that nobody knows.
#
#   BROKEN   named in the copy list, absent from the working tree. These are
#            the lines that print a warning and, before the fatal-miss patch,
#            let the build exit 0 anyway.
#
#   ORPHAN   present in an existing artifact directory, not produced by the
#            current list. These are what `rsync --delete` would remove.
#
#   Usage:  bash audit_artifact_coverage.sh [artifact_dir]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

#   The working tree and the published git clone are separate directories
#   here, so SRC defaults to wherever make_artifact.sh lives and ART is the
#   clone that gets pushed. GAP needs SRC under git; when it is not, that
#   section is skipped and BROKEN and ORPHAN still run. ORPHAN is the one
#   that answers "is rsync --delete safe", and it needs no git at all.
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ART="${1:-}"
if [[ "${1:-}" == "--src" ]]; then
    SRC="$(cd "$2" && pwd)"; ART="${3:-}"
fi
SCRIPT="$SRC/make_artifact.sh"

[[ -f "$SCRIPT" ]] || {
    echo "no make_artifact.sh in $SRC" >&2
    echo "pass the tree that holds it:  bash $0 --src <dir> [artifact_dir]" >&2
    exit 1
}
HAVE_GIT=1
git -C "$SRC" rev-parse --git-dir >/dev/null 2>&1 || HAVE_GIT=0

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# the copy list: `copy <path> [req] [# comment]`, skipping commented-out lines
grep -E '^[[:space:]]*copy[[:space:]]+' "$SCRIPT" \
  | sed -E 's/^[[:space:]]*copy[[:space:]]+([^[:space:]]+).*/\1/' \
  | sort -u > "$TMP/listed"

# editor and patch backups are never publication candidates
if (( HAVE_GIT )); then
    git -C "$SRC" ls-files \
      | grep -Ev '\.(bak[0-9]*|orig|rej|swp)$|~$' \
      | sort -u > "$TMP/tracked"
else
    : > "$TMP/tracked"
fi

# written by make_artifact.sh itself rather than copied, so not an orphan
GENERATED=".gitignore"

# a listed directory publishes everything under it, so expand before diffing
: > "$TMP/covered"
while read -r rel; do
    if [[ -d "$SRC/$rel" ]]; then
        ( cd "$SRC/$rel" && find . -type f -printf "$rel/%P\n" ) >> "$TMP/covered"
    else
        echo "$rel" >> "$TMP/covered"
    fi
done < "$TMP/listed"
sort -u -o "$TMP/covered" "$TMP/covered"

echo "source tree     : $SRC"
echo "artifact / clone: ${ART:-<not given>}"
echo "copy list       : $(wc -l < "$TMP/listed") entries"
echo "covered by list : $(wc -l < "$TMP/covered")"
(( HAVE_GIT )) && echo "tracked in git  : $(wc -l < "$TMP/tracked")"

# ── GAP ─────────────────────────────────────────────────────────────────────
comm -23 "$TMP/tracked" "$TMP/covered" > "$TMP/gap"
n_gap=$(wc -l < "$TMP/gap")

echo ""
if (( ! HAVE_GIT )); then
    n_gap=0
    echo "══ GAP: skipped ══════════════════════════════════════════════════"
    echo "  $SRC is not a git repository, so there is no tracked-file list to"
    echo "  diff against. ORPHAN below is the check that matters for"
    echo "  rsync --delete, and it does not need git."
elif [[ "$n_gap" -eq 0 ]]; then
    echo "══ GAP: tracked in git, not published (0) ════════════════════════"
    echo "  none"
else
    echo "══ GAP: tracked in git, not published ($n_gap) ═══════════════════════"
    # group by top-level directory so the shape of the gap is visible
    awk -F/ '{ print (NF>1 ? $1 : ".") }' "$TMP/gap" | sort | uniq -c \
      | sort -rn | awk '{ printf "  %4d  %s/\n", $1, $2 }'
    echo ""
    echo "  full list:"
    sed 's/^/    /' "$TMP/gap"
fi

# ── BROKEN ──────────────────────────────────────────────────────────────────
: > "$TMP/broken"
while read -r rel; do
    [[ -e "$SRC/$rel" ]] || echo "$rel" >> "$TMP/broken"
done < "$TMP/listed"
n_broken=$(wc -l < "$TMP/broken")

echo ""
echo "══ BROKEN: in the copy list, not in the working tree ($n_broken) ═════"
if [[ "$n_broken" -eq 0 ]]; then
    echo "  none"
else
    sed 's/^/    /' "$TMP/broken"
    echo ""
    echo "  Lines marked 'req' here are the ones that abort the build."
    echo "  Lines without 'req' are copied when present and skipped when not,"
    echo "  so a reader may or may not receive them."
fi

# ── ORPHAN ──────────────────────────────────────────────────────────────────
if [[ -n "$ART" ]]; then
    if [[ -d "$ART" ]]; then
        ( cd "$ART" && find . -type f \
            -not -path './.git/*' -not -name '*.pyc' \
            -not -path '*/__pycache__/*' -printf '%P\n' ) \
            | grep -Fxv "$GENERATED" | sort -u > "$TMP/artifact"
        comm -23 "$TMP/artifact" "$TMP/covered" > "$TMP/orphan"
        n_orph=$(wc -l < "$TMP/orphan")
        echo ""
        echo "══ ORPHAN: in $ART, not produced by the list ($n_orph) ════"
        if [[ "$n_orph" -eq 0 ]]; then
            echo "  none -- rsync --delete is safe against this artifact"
        else
            sed 's/^/    /' "$TMP/orphan"
            echo ""
            echo "  rsync --delete would remove these. Add each to the copy"
            echo "  list, or confirm it is genuinely stale, before using it."
        fi
    else
        echo ""
        echo "  (no directory at $ART, skipping the orphan check)"
    fi
else
    echo ""
    echo "  (pass an artifact directory to also check for orphans)"
fi

echo ""
if [[ "$n_gap" -eq 0 && "$n_broken" -eq 0 ]]; then
    echo "copy list and source tree agree."
else
    [[ "$n_gap" -gt 0 ]] && echo "Decide each GAP entry: publish it, or record why it is withheld."
    [[ "$n_broken" -gt 0 ]] && echo "Each BROKEN entry is a copy-list line naming a file that is not there."
    echo "The artifact is reproducible only if a reader has what the paper"
    echo "says they have."
fi