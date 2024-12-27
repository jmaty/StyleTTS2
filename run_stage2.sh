#!/bin/bash
set -e

export LC_NUMERIC="en_US.UTF-8"

# -----------------------------------------------------------------------------
# INPUT ARGUMENTS
# -----------------------------------------------------------------------------
# Default params
SPEC="gpu3"
RUNS=1
HOURS=24
MODELS=""
INTB=Train_second.ipynb
# QSUB ARGUMENTS
MEM=64gb
LSCRATCH=20gb
NCPUS=8
NGPUS=2

if [[ "$#" -lt 1 ]]; then
     echo "Usage: run_stage1.sh exp_dir [specification: iti dgx gpu<3-4>] [hours]"
     exit 1
fi

# Input experimental directory
EXPDIR=$1

if [[ "$#" -gt 1 ]]; then
     # specification to run on (iti, gdx, gpu<3-4>)
     SPEC=$2
fi
if [[ "$#" -gt 2 ]]; then
     # Number of hours
     HOURS=$3
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
     echo "Unsupported cluster/queue"
     exit 1
fi

# Change GPU queue to gpu_long when number of hours is >24
[[ $HOURS -gt 24 ]] && [[ $SPEC == gpu? ]] && QUEUE="${QUEUE}_long"

# Select argument
SELECT="-l select=1:ncpus=$NCPUS:mem=$MEM:scratch_local=$LSCRATCH:ngpus=$NGPUS$CLUSTER"
# Walltime argument
WALLTIME="-l walltime=$HOURS:00:00"

# Extract name of the experiment
EXP="$(basename $EXPDIR)_stage2"

# Timestep to differentiate among runs with the same run name
TIMESTEP=$(date +"%y%m%d-%H%M%S")

SINGULARITY=/storage/plzen4-ntis/home/jmatouse/singularity/papermill_23.12-latest.sh

# Check that config file exists
CFG=$EXPDIR/config2.yml
if [[ ! -e $CFG ]]; then
     echo "Config file $CFG does not exists!"
     exit 1
fi

# Set the log dir according to the input experiment directory
# (the original log dir in the config file serves just as a placeholder)
sed -i "/^log_dir:/c\log_dir: $EXPDIR" $CFG

# -----------------------------------------------------------------------------
# RUN TRAINING
# -----------------------------------------------------------------------------
OLOG=$EXPDIR/stage2.$TIMESTEP.log
ONTB=$EXPDIR/$(basename "$INTB" .ipynb).processed.$TIMESTEP.ipynb

# Run PBS script
qsub -N "$EXP" \
     $QUEUE \
     -j oe \
     -o $OLOG \
     $WALLTIME \
     $SELECT \
     -- $SINGULARITY "$INTB" "$CFG" "$ONTB"
echo "$EXP: $QUEUE $SELECT, HOURS: $HOURS"
