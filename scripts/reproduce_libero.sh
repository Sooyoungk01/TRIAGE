#!/bin/bash
# usage: scripts/reproduce_libero.sh <openvla|openvla-oft>
set -e
cd "$(dirname "$0")/../src/$1"; shift
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  for START in 0 25; do
    python experiments/robot/libero/run_libero_eval.py --task_suite_name $SUITE --trial_start $START --num_trials_per_task 25 "$@"
  done
done
