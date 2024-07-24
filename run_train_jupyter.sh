#!/bin/bash
set -e

export LC_NUMERIC="en_US.UTF-8"

# -----------------------------------------------------------------------------
# INPUT ARGUMENTS
# -----------------------------------------------------------------------------
# Default params
JOBID=""
SPEC="gpu3"
RUNS=1
HOURS=24
MODELS=""
# QSUB ARGUMENTS
MEM=64gb
LSCRATCH=20gb

if [[ "$#" -lt 2 ]]; then
     echo "Usage: run_train_jupyter.sh cfg notebook [specification: iti dgx gpu<0-3>] [hours] [runs] [jobid]"
     exit 1
fi
if [[ "$#" -gt 1 ]]; then
     # configuration file
     CFG=$1
     # I/O Jupyter notebook
     INTB=$2
fi
if [[ "$#" -gt 2 ]]; then
     # specification to run on (iti, gdx, gpu<0-3>)
     SPEC=$3
fi
if [[ "$#" -gt 3 ]]; then
     # Number of hours
     HOURS=$4
fi
if [[ "$#" -gt 4 ]]; then
     #  Number of runs
     RUNS=$5
fi
if [[ "$#" -gt 5 ]]; then
     # JOBID to continue run
     JOBID=$6
fi

# Check dependencies
if [[ -z $JOBID ]]; then
     # No deps at the beginning
     DEPS=""
     PREV_JOBID=""
else
     DEPS="-W depend=afterany:$JOBID"
     PREV_JOBID=$JOBID
fi

# Check run specification and set queue and cluster to run on
if [[ $SPEC == "iti0" || $SPEC == "iti"  ]]; then
     # ITI queue: any machine
     QUEUE="-q iti"
     CLUSTER=""
elif [[ $SPEC == "iti1" ]]; then
     # ITI queue: alfrid (>40 gb GPU)
     QUEUE="-q iti"
     CLUSTER=":gpu_mem=40000mb"
elif [[ $SPEC == "dgx" ]]; then
     # GDX queue
     QUEUE="-q gpu_dgx"
     CLUSTER=""
elif [[ $SPEC == "gpu0" || $SPEC == "gpu" ]]; then
     # Any cluster with gpu (incl. problematic zubat, gita)
     QUEUE="-q gpu"
     CLUSTER=""
     SPEC="gpu0"
elif [[ $SPEC == "gpu1" ]]; then
     # Any cluster with GPU memory > 11gb (glados)
     QUEUE="-q gpu"
     CLUSTER=":gpu_mem=11111mb"
elif [[ $SPEC == "gpu2" ]]; then
     # Any cluster with GPU memory > 12gb (excl. glados)
     QUEUE="-q gpu"
     CLUSTER=":gpu_mem=12000mb"
elif [[ $SPEC == "gpu3" ]]; then
     # Any cluster with GPU memory > 40gb (zia, black)
     QUEUE="-q gpu"
     CLUSTER=":gpu_mem=40000mb"
else
     echo "Unsupported cluster/queue"
     exit 1
fi

# Change GPU queue to gpu_long when number of hours is >24
[[ $HOURS -gt 24 ]] && [[ $SPEC == gpu? ]] && QUEUE="${QUEUE}_long"

# Select argument
SELECT="-l select=1:ncpus=2:mem=$MEM:scratch_local=$LSCRATCH:ngpus=1$CLUSTER"
# Walltime argument
WALLTIME="-l walltime=$HOURS:00:00"

# Extract name of the experiment
EXP=$(grep 'log_dir:' "$CFG" | sed -r 's/log_dir: "Models\/(.+)"/\1/')

# Timestep to differentiate among runs with the same run name
TIMESTEP=$(date +"%y%m%d_%H%M%S")

SINGULARITY=/storage/plzen4-ntis/projects/singularity/papermill_23.12-latest.sh

# -----------------------------------------------------------------------------
# RUN TRAINING
# -----------------------------------------------------------------------------

# Run the script sequentially with the given number of repetions `RUNS`
for i in $(seq 1 $RUNS); do
     ONTB=output_notebooks/$(basename "$INTB" .ipynb)_"$EXP"-$i.$TIMESTEP.ipynb
     # Run PBS script
     JOBID=$(qsub -N "$EXP" \
          $QUEUE \
          -j oe \
          -o output_logs/$EXP.$TIMESTEP.log \
          $WALLTIME \
          $SELECT \
          $DEPS \
          -- $SINGULARITY "$INTB" "$CFG" "$ONTB")
     echo "$EXP: $QUEUE$CLUSTER, RUN: $i, HOURS: $HOURS, JOBID: $PREV_JOBID -> $JOBID"
     # Update dependencies to enable sequential run
     DEPS="-W depend=afterany:$JOBID"
     PREV_JOBID=$JOBID
done
