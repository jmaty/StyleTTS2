#!/bin/bash
set -e

export LC_NUMERIC="en_US.UTF-8"

# -----------------------------------------------------------------------------
# INPUT ARGUMENTS
# -----------------------------------------------------------------------------
# Default params
SPEC="gpu3"
HOURS=72
INTB=Train_second.ipynb
# QSUB ARGUMENTS
MEM=256gb
LSCRATCH=20gb
SCRATCH_TYPE="scratch-local"
NCPUS1=16  # Number of CPUs per process (GPU) for DP computing
NGPUS=2

if [[ "$#" -lt 1 ]]; then
     printf "Usage: run_stage2b.sh config [specification: iti dgx gpu<3-4>] [hours] [ngpus] [jobid]\n" >&2
     exit 1
fi

CFG=$1
# Check that config file exists
if [[ ! -e $CFG ]]; then
     printf "Config file $CFG does not exists!" >&2
     exit 1
fi
# Experimental directory set to the directory of the config file
EXPDIR=$(dirname $CFG)

if [[ "$#" -gt 1 ]]; then
     # specification to run on (iti, gdx, gpu<3-4>)
     SPEC=$2
fi
if [[ "$#" -gt 2 ]]; then
     # Number of hours
     HOURS=$3
fi
if [[ "$#" -gt 3 ]]; then
     # Number of GPUs
     NGPUS=$4
     # Set number of CPUs according to the number of GPUs (for DP computing)
     NCPUS=$((NGPUS * NCPUS1))
fi
if [[ "$#" -gt 4 ]]; then
     # JOBID to continue run
     JOBID=$5
fi

# Check dependencies
if [[ -z $JOBID ]]; then
     # No deps at the beginning
     DEPS=""
else
     DEPS="-W depend=afterok:$JOBID"
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
     SCRATCH_TYPE="scratch_ssd"
elif [[ $SPEC == "gpu3" ]]; then
     # Any cluster with GPU memory > 40gb (zia, black)
     QUEUE="-q gpu"
     CLUSTER=":gpu_mem=40000mb"
elif [[ $SPEC == "gpu4" ]]; then
     # Any cluster with GPU memory > 80gb (bee)
     QUEUE="-q gpu"
     CLUSTER=":gpu_mem=80000mb"
     SCRATCH_TYPE="scratch_ssd"
else
     printf "Unsupported cluster/queue" >&2
     exit 1
fi

# Change GPU queue to gpu_long when number of hours is >48
[[ $HOURS -gt 48 ]] && [[ $SPEC == gpu? ]] && QUEUE="${QUEUE}_long"

# Select argument
SELECT="-l select=1:ncpus=$NCPUS:mem=$MEM:$SCRATCH_TYPE=$LSCRATCH:ngpus=$NGPUS$CLUSTER"
# Walltime argument
WALLTIME="-l walltime=$HOURS:00:00"

# Prepare name of the run: config2a.yml -> 2a
BASENAME_CFG=$(basename "$CFG")
TEMP_RUN="${BASENAME_CFG#config}"  # Remove 'config' prefix
RUN="${TEMP_RUN%.*}"               # Remove file extension
EXP=$(basename $EXPDIR)_$RUN  # Set name of the experiment

# Timestep to differentiate among runs with the same run name
TIMESTEP=$(date +"%y%m%d-%H%M%S")

SINGULARITY=/storage/plzen4-ntis/projects/singularity/papermill_24.12-r8.sh

# Set the log dir according to the input experiment directory
# (the original log dir in the config file serves just as a placeholder)
sed -i "/^log_dir:/c\log_dir: $EXPDIR" $CFG

# Transfer sigma_data between runs
# - config2.processed.yml was created in previous run
if [[ -e $EXPDIR/config2.processed.yml ]]; then     
     sigma_data=$(grep -E '^[[:space:]]*sigma_data:' $EXPDIR/config2.processed.yml)
     sed -i "/^[[:space:]]*sigma_data:/c\\$sigma_data" $CFG
fi

# -----------------------------------------------------------------------------
# RUN TRAINING
# -----------------------------------------------------------------------------
OLOG=$EXPDIR/stage$RUN.$TIMESTEP.log
ONTB=$EXPDIR/stage$RUN.$TIMESTEP.ipynb

# Run PBS script
JOBID=$(qsub -N $EXP \
     $QUEUE \
     -j oe \
     -o $OLOG \
     $WALLTIME \
     $SELECT \
     $DEPS \
     -- $SINGULARITY "$INTB" "$CFG" "$ONTB")
printf "$JOBID"
# echo "$EXP: $QUEUE $SELECT, HOURS: $HOURS <-- $JOBID"
