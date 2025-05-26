#!/bin/bash
set -e

export LC_NUMERIC="en_US.UTF-8"

# -----------------------------------------------------------------------------
# INPUT ARGUMENTS
# -----------------------------------------------------------------------------
if [[ "$#" -lt 1 ]]; then
     echo "Usage: run_all.sh exp_dir [start_stage] [previous_jobid]" >&2
     echo "Files settings.yml and config.yml must be in the exp_dir" >&2
     echo "Valid start_stages: 1a, 1b, 2a, 2b, 2c. Default: 1a" >&2
     exit 1
fi

EXPDIR=$1  # Experimental directory
START_STAGE="${2:-1a}" # Druhý argument pro startovací fázi, defaultně "1a"
PREV_JOBID="${3:-}" # Třetí argument pro ID předchozí úlohy, defaultně prázdný

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

# Validace start_stage
case "$START_STAGE" in
    1a|1b|2a|2b|2c)
        echo "Requested start stage: $START_STAGE"
        ;;
    *)
        echo "Error: Invalid start_stage '$START_STAGE'." >&2
        echo "Valid start_stages: 1a, 1b, 2a, 2b, 2c." >&2
        exit 1
        ;;
esac

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

# Initialize job IDs
jobid1a=""
jobid1b=""
jobid2a=""
jobid2b=""
jobid2c=""

can_run=false # Příznak pro spuštění fáze

# -----------------------------------------------------------------------------
# Run stage 1a
[[ "$START_STAGE" == "1a" ]] && can_run=true
if [[ "$can_run" == true ]]; then
    ngpus=$(yq '.start1.ngpus' "$SETTINGS")
    [[ $ngpus -gt 2 ]] && queue=dgx || queue=gpu4
    hours=$(yq '.start1.hours' "$SETTINGS")
    jobid1a=$(./run_stage1.sh $EXPDIR/${CONFIG_NAME}1a.yml $queue $hours $ngpus 2>/dev/null)
    echo "Stage 1a submitted. Job ID: ${jobid1a}"
fi

# Run stage 1b
[[ "$START_STAGE" == "1b" ]] && can_run=true
if [[ "$can_run" == true ]]; then
    ngpus=$(yq '.tma.ngpus' "$SETTINGS")
    [[ $ngpus -gt 2 ]] && queue=dgx || queue=gpu4
    hours=$(yq '.tma.hours' "$SETTINGS")
    current_dep=$jobid1a # Standardní závislost
    if [[ "$START_STAGE" == "1b" && -n "$PREV_JOBID" ]]; then
        current_dep="$PREV_JOBID"
        echo "Using provided job ID ($PREV_JOBID) for stage 1b dependency."
    fi
    jobid1b=$(./run_stage1.sh $EXPDIR/${CONFIG_NAME}1b.yml $queue $hours $ngpus $current_dep 2>/dev/null)
    echo "Stage 1b submitted. Job ID: ${jobid1b}"
fi

# Run stage 2a
[[ "$START_STAGE" == "2a" ]] && can_run=true
if [[ "$can_run" == true ]]; then
    ngpus=$(yq '.start2.ngpus' "$SETTINGS")
    [[ $ngpus -gt 2 ]] && queue=dgx || queue=gpu4
    hours=$(yq '.start2.hours' "$SETTINGS")
    current_dep=$jobid1b # Standardní závislost
    if [[ "$START_STAGE" == "2a" && -n "$PREV_JOBID" ]]; then
        current_dep="$PREV_JOBID"
        echo "Using provided job ID ($PREV_JOBID) for stage 2a dependency."
    fi
    jobid2a=$(./run_stage2.sh $EXPDIR/${CONFIG_NAME}2a.yml $queue $hours $ngpus $current_dep 2>/dev/null)
    echo "Stage 2a submitted. Job ID: ${jobid2a}"
fi

# Run stage 2b
[[ "$START_STAGE" == "2b" ]] && can_run=true
if [[ "$can_run" == true ]]; then
    ngpus=$(yq '.diff.ngpus' "$SETTINGS")
    [[ $ngpus -gt 2 ]] && queue=dgx || queue=gpu4
    hours=$(yq '.diff.hours' "$SETTINGS")
    current_dep=$jobid2a # Standardní závislost
    if [[ "$START_STAGE" == "2b" && -n "$PREV_JOBID" ]]; then
        current_dep="$PREV_JOBID"
        echo "Using provided job ID ($PREV_JOBID) for stage 2b dependency."
    fi
    jobid2b=$(./run_stage2.sh $EXPDIR/${CONFIG_NAME}2b.yml $queue $hours $ngpus $current_dep 2>/dev/null)
    echo "Stage 2b submitted. Job ID: ${jobid2b}"
fi

# Run stage 2c
[[ "$START_STAGE" == "2c" ]] && can_run=true
if [[ "$can_run" == true ]]; then
    ngpus=$(yq '.joint.ngpus' "$SETTINGS")
    [[ $ngpus -gt 2 ]] && queue=dgx || queue=gpu4
    hours=$(yq '.joint.hours' "$SETTINGS")
    current_dep=$jobid2b # Standardní závislost
    if [[ "$START_STAGE" == "2c" && -n "$PREV_JOBID" ]]; then
        current_dep="$PREV_JOBID"
        echo "Using provided job ID ($PREV_JOBID) for stage 2c dependency."
    fi
    jobid2c=$(./run_stage2.sh $EXPDIR/${CONFIG_NAME}2c.yml $queue $hours $ngpus $current_dep 2>/dev/null)
    echo "Stage 2c submitted. Job ID: ${jobid2c}"
fi
