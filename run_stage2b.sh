#!/bin/bash
set -e

export LC_NUMERIC="en_US.UTF-8"

# -----------------------------------------------------------------------------
# INPUT ARGUMENTS
# -----------------------------------------------------------------------------
# Default params
SPEC="gpu3"
RUNS=1
HOURS=72
MODELS=""
INTB=Train_second.ipynb
# QSUB ARGUMENTS
MEM=64gb
LSCRATCH=20gb
NCPUS=8
NGPUS=2

if [[ "$#" -lt 1 ]]; then
     printf "Usage: run_stage2a.sh exp_dir [specification: iti dgx gpu<3-4>] [hours] [jobid]" >&2
     exit 1
fi

# Input experimental directory
EXPDIR=$1
CFG=$EXPDIR/config2b.yml

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
[[ $HOURS -gt 24 ]] && [[ $SPEC == gpu? ]] && QUEUE="${QUEUE}_long"

# Select argument
SELECT="-l select=1:ncpus=$NCPUS:mem=$MEM:scratch_local=$LSCRATCH:ngpus=$NGPUS$CLUSTER"
# Walltime argument
WALLTIME="-l walltime=$HOURS:00:00"

# Extract name of the experiment
EXP="$(basename $EXPDIR)_stage2b"

# Timestep to differentiate among runs with the same run name
TIMESTEP=$(date +"%y%m%d-%H%M%S")

SINGULARITY=/storage/plzen4-ntis/home/jmatouse/singularity/papermill_23.12-latest.sh

# Check that config file exists
if [[ ! -e $CFG ]]; then
     printf "Config file $CFG does not exists!" >&2
     exit 1
fi

# Set the log dir according to the input experiment directory
# (the original log dir in the config file serves just as a placeholder)
sed -i "/^log_dir:/c\log_dir: $EXPDIR" $CFG
# Set the pretrained model path
sed -i "/^pretrained_model:/c\pretrained_model: $EXPDIR/stage2_pre-joint_00049.pth" $CFG
# # Transfer sigma_data from stage2a to stage2b
# sigma_data=$(grep -E '^[[:space:]]*sigma_data:' $EXPDIR/config2.processed.yml)
# sed -i "/^[[:space:]]*sigma_data:/c\\$sigma_data" $CFG

# -----------------------------------------------------------------------------
# RUN TRAINING
# -----------------------------------------------------------------------------
OLOG=$EXPDIR/stage2b.$TIMESTEP.log
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
printf "$JOBID"
# echo "$EXP: $QUEUE $SELECT, HOURS: $HOURS <-- $JOBID"
