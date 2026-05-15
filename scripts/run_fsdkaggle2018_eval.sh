#!/bin/bash
set -e

# Source environment setup
source "$(dirname "$0")/env_setup.sh" "$@"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

echo "Starting evaluation for fsdkaggle2018..."

# Run xares.run using uv
# --max-jobs 1: Safer for single GPU execution
# --from-stage 0: Start from download
uv run python -u -m xares.run \
    src/xares/encoders/audio_jepa.py \
    src/tasks/fsdkaggle2018_task.py \
    --max-jobs 1 \
    --from-stage 0 2> >(grep -v "DeprecationWarning: \`TorchScript\`" >&2)

echo "Evaluation for fsdkaggle2018 completed."
