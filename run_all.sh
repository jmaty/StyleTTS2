#!/bin/bash
set -e

export LC_NUMERIC="en_US.UTF-8"

# -----------------------------------------------------------------------------
# INPUT ARGUMENTS
# -----------------------------------------------------------------------------
if [[ "$#" -lt 1 ]]; then
     echo "Usage: run_all.sh exp_dir" >&2
     echo "Files settings.yml and config.yml must be in the exp_dir" >&2
     exit 1
fi

EXPDIR=$1  # Experimental directory
if [[ ! -d $EXPDIR ]]; then
     echo "Experimental directory $EXPDIR does not exist!" >&2
     exit 1
fi

SETTINGS=$EXPDIR/settings.yml  # Training stages settings file
CONFIG=$EXPDIR/config.yml  # Experiment config file

if [[ ! -e $SETTINGS ]]; then
     echo "Settings file $SETTINGS does not exist!" >&2
     exit 1
fi
if [[ ! -e $CONFIG ]]; then
     echo "Config file $CONFIG does not exist!" >&2
     exit 1
fi

CONFIG_NAME=$(basename "$CONFIG" .yml)

# Prepare configs for various training phases
# - config1a.yml: stage 1a starting from scratch
# - config1b.yml: stage 1b starting from TMA epoch
# - config2a.yml: stage 2a starting from scratch
# - config2b.yml: stage 2a starting from diff epoch
# - config2c.yml: stage 2b starting from joint epoch
./Bin/prep_configs.py $SETTINGS $CONFIG

# Notes on settings (read from the settings file):
# - ngpus: number of GPUs to use. If > 2, use dgx queue (capy), else gpu4 (bee)
# - hours: number of hours to run the job

# Note on training phases (read from the settings file):
# - start1: stage 1a (starting from scratch)
# - tma: stage 1b (starting from TMA epoch)
# - start2: stage 2a (starting from scratch)
# - diff: stage 2b (starting from diff epoch)
# - joint: stage 2c (starting from joint epoch)

# -----------------------------------------------------------------------------
# Run stage 1a
ngpus=$(yq '.start1.ngpus' "$SETTINGS")
[[ $ngpus -gt 2 ]] && queue=dgx || queue=gpu4
hours=$(yq '.start1.hours' "$SETTINGS")
jobid1a=$(./run_stage1.sh $EXPDIR/${CONFIG_NAME}1a.yml $queue $hours $ngpus 2>/dev/null)

# Run stage 1b
ngpus=$(yq '.tma.ngpus' "$SETTINGS")
[[ $ngpus -gt 2 ]] && queue=dgx || queue=gpu4
hours=$(yq '.tma.hours' "$SETTINGS")
jobid1b=$(./run_stage1.sh $EXPDIR/${CONFIG_NAME}1b.yml $queue $hours $ngpus $jobid1a 2>/dev/null)

# Run stage 2a
ngpus=$(yq '.start2.ngpus' "$SETTINGS")
[[ $ngpus -gt 2 ]] && queue=dgx || queue=gpu4
hours=$(yq '.start2.hours' "$SETTINGS")
jobid2a=$(./run_stage2.sh $EXPDIR/${CONFIG_NAME}2a.yml $queue $hours $ngpus $jobid1b 2>/dev/null)

# Run stage 2b
ngpus=$(yq '.diff.ngpus' "$SETTINGS")
[[ $ngpus -gt 2 ]] && queue=dgx || queue=gpu4
hours=$(yq '.diff.hours' "$SETTINGS")
jobid2b=$(./run_stage2.sh $EXPDIR/${CONFIG_NAME}2b.yml $queue $hours $ngpus $jobid2a 2>/dev/null)

# Run stage 2c
ngpus=$(yq '.joint.ngpus' "$SETTINGS")
[[ $ngpus -gt 2 ]] && queue=dgx || queue=gpu4
hours=$(yq '.joint.hours' "$SETTINGS")
jobid2c=$(./run_stage2.sh $EXPDIR/${CONFIG_NAME}2c.yml $queue $hours $ngpus $jobid2b 2>/dev/null)

printf "$jobid1a -> $jobid1b -> $jobid2a -> $jobid2b -> $jobid2c\n"
