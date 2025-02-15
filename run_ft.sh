#!/bin/bash
set -e

export LC_NUMERIC="en_US.UTF-8"

# -----------------------------------------------------------------------------
# INPUT ARGUMENTS
# -----------------------------------------------------------------------------
# Default params
SPEC="gpu3"
HOURS=72
INTB=Train_finetune.ipynb
# QSUB ARGUMENTS
MEM=128gb
LSCRATCH=20gb
NCPUS=8
NGPUS=1

if [[ "$#" -lt 1 ]]; then
     printf "Usage: run_ft.sh exp_dir [specification: iti dgx gpu<3-4>] [hours] [jobid]\n" >&2
     exit 1
fi

# Input experimental directory
EXPDIR=$1
# Check that config file exists
CFG=$EXPDIR/config.yml
if [[ ! -e $CFG ]]; then
     printf "Config file $CFG does not exists!\n" >&2
     exit 1
fi

if [[ "$#" -gt 1 ]]; then
     # specification to run on (iti, gdx, gpu<3-4>)
     SPEC=$2
fi
if [[ "$#" -gt 2 ]]; then
     # Number of hours
     HOURS=$3
fi
if [[ "$#" -gt 3 ]]; then
     # JOBID to continue run
     JOBID=$4
fi

# Check dependencies
if [[ -z $JOBID ]]; then
     # No deps at the beginning
     DEPS=""
else
     DEPS="-W depend=afterany:$JOBID"
fi

# Check run specification and set queue and cluster to run on
if [[ $SPEC == "iti" ]]; then
     # ITI queue: alfrid (>40 gb GPU)
     QUEUE="-q iti"
     CLUSTER=":gpu_mem=40000mb"
elif [[ $SPEC == "dgx" ]]; then
     # GDX queue: capy
     QUEUE="-q gpu_dgx"
     CLUSTER=""
elif [[ $SPEC == "gpu3" ]]; then
     # Any cluster with GPU memory > 40gb (zia, black)
     QUEUE="-q gpu"
     CLUSTER=":gpu_mem=40000mb"
elif [[ $SPEC == "gpu4" ]]; then
     # Any cluster with GPU memory > 80gb (bee)
     QUEUE="-q gpu"
     CLUSTER=":gpu_mem=80000mb"
else
     printf "Unsupported cluster/queue" >&2
     exit 1
fi

# Change GPU queue to gpu_long when number of hours is >24
[[ $HOURS -gt 48 ]] && [[ $SPEC == gpu? ]] && QUEUE="${QUEUE}_long"

# Select argument
SELECT="-l select=1:ncpus=$NCPUS:mem=$MEM:scratch_local=$LSCRATCH:ngpus=$NGPUS$CLUSTER"
# Walltime argument
WALLTIME="-l walltime=$HOURS:00:00"

# Extract name of the experiment
EXP="$(basename $EXPDIR)_ft"

# Timestep to differentiate among runs with the same run name
TIMESTEP=$(date +"%y%m%d-%H%M%S")

SINGULARITY=/storage/plzen4-ntis/home/jmatouse/singularity/papermill_23.12-latest.sh

# Set the log dir according to the input experiment directory
# (the original log dir in the config file serves just as a placeholder)
sed -i "/^log_dir:/c\log_dir: $EXPDIR" $CFG
# Transfer sigma_data from stage2a to stage2b
if [[ -e $EXPDIR/config.processed.yml ]]; then     
     sigma_data=$(grep -E '^[[:space:]]*sigma_data:' $EXPDIR/config.processed.yml)
     sed -i "/^[[:space:]]*sigma_data:/c\\$sigma_data" $CFG
fi

# -----------------------------------------------------------------------------
# RUN TRAINING
# -----------------------------------------------------------------------------
OLOG=$EXPDIR/ft.$TIMESTEP.log
ONTB=$EXPDIR/$(basename "$INTB" .ipynb).processed.$TIMESTEP.ipynb

# Run PBS script
JOBID=$(qsub -N "$EXP" \
     $QUEUE \
     -j oe \
     -o $OLOG \
     $WALLTIME \
     $SELECT \
     $DEPS \
     -- $SINGULARITY "$INTB" "$CFG" "$ONTB")
printf "$JOBID\n"
# echo "$EXP: $QUEUE $SELECT, HOURS: $HOURS <-- $JOBID"
