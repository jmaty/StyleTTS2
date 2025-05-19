#!/bin/bash
set -e

export LC_NUMERIC="en_US.UTF-8"

# -----------------------------------------------------------------------------
# INPUT ARGUMENTS
# -----------------------------------------------------------------------------
if [[ "$#" -lt 1 ]]; then
     echo "Usage: run_all.sh settings.yml" >&2
     exit 1
fi
SETTINGS=$1
CONFIG=$2

# Input experimental directory
EXPDIR=$(dirname "$CFG")

# Prepare configs for various training phases
# - config1a.yml: stage 1 starting from scratch
# - config1b.yml: stage 1 starting from TMA epoch
# - config2a.yml: stage 2a starting from scratch
# - config2b.yml: stage 2a starting from diff epoch
# - config2c.yml: stage 2b starting from joint epoch
./Bin/prep_configs.sh $SETTINGS config.yml

# Run stage 1a
ngpus=$(yq '.start1.ngpus' "$SETTINGS")
[[ $ngpus -gt 2 ]] && queue=dgx || queue=gpu4
hours=$(yq '.start1.hours' "$SETTINGS")
jobid1a=$(./run_stage1.sh $EXPDIR/config1a.yml $queue $hours $ngpus 2>/dev/null)

# Run stage 1b
ngpus=$(yq '.tma.ngpus' "$SETTINGS")
[[ $ngpus -gt 2 ]] && queue=dgx || queue=gpu4
hours=$(yq '.tma.hours' "$SETTINGS")
jobid1b=$(./run_stage1.sh $EXPDIR/config1b.yml $queue $hours $ngpus 2>/dev/null)

# Run stage 2a
ngpus=$(yq '.start2.ngpus' "$SETTINGS")
[[ $ngpus -gt 2 ]] && queue=dgx || queue=gpu4
hours=$(yq '.start2.hours' "$SETTINGS")
jobid2a=$(./run_stage2.sh $EXPDIR/config2a.yml $queue $hours $ngpus 2>/dev/null)

# Run stage 2b
ngpus=$(yq '.diff.ngpus' "$SETTINGS")
[[ $ngpus -gt 2 ]] && queue=dgx || queue=gpu4
hours=$(yq '.diff.hours' "$SETTINGS")
jobid2b=$(./run_stage2.sh $EXPDIR/config2b.yml $queue $hours $ngpus 2>/dev/null)

# Run stage 2c
ngpus=$(yq '.joint.ngpus' "$SETTINGS")
[[ $ngpus -gt 2 ]] && queue=dgx || queue=gpu4
hours=$(yq '.joint.hours' "$SETTINGS")
jobid2c=$(./run_stage2.sh $EXPDIR/config2c.yml $queue $hours $ngpus 2>/dev/null)

printf "$jobid1a -> $jobid1b -> $jobid2a -> $jobid2b -> $jobid2c\n"
