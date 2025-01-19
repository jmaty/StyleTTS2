#!/bin/bash
set -e

export LC_NUMERIC="en_US.UTF-8"

# -----------------------------------------------------------------------------
# INPUT ARGUMENTS
# -----------------------------------------------------------------------------
# Default params
QUEUE="dgx"
HOURS1=60
HOURS2a=30
HOURS2b=72
NGPUS1=2
NGPUS2a=2
NGPUS2b=2

if [[ "$#" -lt 1 ]]; then
     echo "Usage: run_capy.sh exp_dir [hours1=$HOURS1] [hours2a=$HOURS2a] [hours2b=$HOURS2b] [ngpus1=$NGPUS1] [ngpus2a=$NGPUS2a] [ngpus2b=$NGPUS2b]" >&2
     exit 1
fi
# Input experimental directory
EXPDIR=$1

if [[ "$#" -gt 1 ]]; then
     HOURS1=$2
fi
if [[ "$#" -gt 2 ]]; then
     HOURS2a=$3
fi
if [[ "$#" -gt 3 ]]; then
     HOURS2b=$4
fi

# Run stage 1
jobid1=$(./run_stage1.sh $EXPDIR $QUEUE $HOURS1 2>/dev/null)
# Run stage 2a
jobid2a=$(./run_stage2a.sh $EXPDIR $QUEUE $HOURS2a $jobid1 2>/dev/null)
# Run stage 2b
jobid2b=$(./run_stage2b.sh $EXPDIR $QUEUE $HOURS2b $jobid2a 2>/dev/null)

printf "$jobid1 -> $jobid2a -> $jobid2b\n"
