#!/bin/bash
set -e

export LC_NUMERIC="en_US.UTF-8"

# -----------------------------------------------------------------------------
# INPUT ARGUMENTS
# -----------------------------------------------------------------------------
if [[ "$#" -lt 1 ]]; then
     echo "Usage: run_stages.sh exp_dir [start_stage] [previous_jobid]" >&2
     echo "Files settings.yml and config.yml must be in the exp_dir" >&2
     echo "Valid start_stages: 1, 2. Default: 1" >&2
     exit 1
fi

EXPDIR=$1  # Experimental directory
START_STAGE="${2:-1}" # Druhý argument pro startovací fázi, defaultně "1"
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
    1|2)
        echo "Requested start stage: $START_STAGE"
        ;;
    *)
        echo "Error: Invalid start_stage '$START_STAGE'." >&2
        echo "Valid start_stages: 1, 2." >&2
        exit 1
        ;;
esac

CONFIG_NAME=$(basename "$CONFIG" .yml)

# Prepare configs for various training phases
# - config1.yml: stage 1
# - config2.yml: stage 2
./Bin/prep_configs.py -s $SETTINGS $CONFIG

# Notes on settings (read from the settings file):
# - ngpus: number of GPUs to use. If > 2, use dgx queue (capy), else gpu4 (bee)
# - hours: number of hours to run the job

# Initialize job IDs
jobid1=""
jobid2=""

can_run=false # Příznak pro spuštění fáze

# -----------------------------------------------------------------------------
# Run stage 1
[[ "$START_STAGE" == "1" ]] && can_run=true
if [[ "$can_run" == true ]]; then
    ngpus=$(yq '.stage1.ngpus' "$SETTINGS")
    [[ $ngpus -gt 2 ]] && queue=dgx || queue=gpu4
    hours=$(yq '.stage1.hours' "$SETTINGS")
    jobid1=$(./run_stage1.sh $EXPDIR/${CONFIG_NAME}1.yml $queue $hours $ngpus 2>/dev/null)
    echo "Stage 1 submitted. Job ID: ${jobid1}"
fi

# -----------------------------------------------------------------------------
# Run stage 2
[[ "$START_STAGE" == "2" ]] && can_run=true
if [[ "$can_run" == true ]]; then
    ngpus=$(yq '.stage2.ngpus' "$SETTINGS")
    [[ $ngpus -gt 2 ]] && queue=dgx || queue=gpu4
    hours=$(yq '.stage2.hours' "$SETTINGS")
    current_dep=$jobid1 # Standardní závislost
    if [[ "$START_STAGE" == "2" && -n "$PREV_JOBID" ]]; then
        current_dep="$PREV_JOBID"
        echo "Using provided job ID ($PREV_JOBID) for stage 2 dependency."
    fi
    jobid2=$(./run_stage2.sh $EXPDIR/${CONFIG_NAME}2.yml $queue $hours $ngpus $current_dep 2>/dev/null)
    echo "Stage 2 submitted. Job ID: ${jobid2}"
fi
# -----------------------------------------------------------------------------
