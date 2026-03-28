#!/bin/bash
# Run v1_7frames part1~part4 in parallel, one GPU per part.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

mkdir -p results

echo "Launching 4 parallel jobs (GPU 0/1/2/3)..."

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_v1_7frames_part1.sh > results/run_v1_part1_gpu0.log 2>&1 &
PID1=$!
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_v1_7frames_part2.sh > results/run_v1_part2_gpu1.log 2>&1 &
PID2=$!
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/run_v1_7frames_part3.sh > results/run_v1_part3_gpu2.log 2>&1 &
PID3=$!
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/run_v1_7frames_part4.sh > results/run_v1_part4_gpu3.log 2>&1 &
PID4=$!

echo "Started:"
echo "  part1 -> GPU0, PID=${PID1}"
echo "  part2 -> GPU1, PID=${PID2}"
echo "  part3 -> GPU2, PID=${PID3}"
echo "  part4 -> GPU3, PID=${PID4}"
echo
echo "Logs:"
echo "  results/run_v1_part1_gpu0.log"
echo "  results/run_v1_part2_gpu1.log"
echo "  results/run_v1_part3_gpu2.log"
echo "  results/run_v1_part4_gpu3.log"
echo
echo "Waiting for all jobs to finish..."

wait "$PID1"
wait "$PID2"
wait "$PID3"
wait "$PID4"

echo "All 4 v1 part jobs completed."
