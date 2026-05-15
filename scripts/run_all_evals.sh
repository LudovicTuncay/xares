#!/bin/bash
set -e

# Source environment setup
source "$(dirname "$0")/env_setup.sh" "$@"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run evaluation on ALL tasks
# We use 'uv run' to execute the module
# encoder_py: Path to the adapter
# tasks_py: Path to the task configuration files (using wildcard)
# --max-jobs 1: Run sequentially (or set higher if multiple GPUs/parallelism desired, but safest with 1)
# --from-stage 0: Start from download

echo "Starting evaluation on all datasets..."

uv run python -u -m xares.run \
    src/xares/encoders/audio_jepa.py \
    src/tasks/*_task.py \
    --max-jobs 1 \
    --from-stage 1 2> >(grep -v "DeprecationWarning: \`TorchScript\`" >&2)

echo "All evaluations completed."
