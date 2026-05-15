#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <log_folder_path>"
    echo "Example: $0 /path/to/logs/train/runs/YYYY-MM-DD_HH-MM-SS"
    exit 1
fi

LOG_FOLDER="$1"

# Remove trailing slash if present
LOG_FOLDER="${LOG_FOLDER%/}"

# Check if log folder exists
if [ ! -d "$LOG_FOLDER" ]; then
    echo "Error: Log folder '$LOG_FOLDER' does not exist."
    exit 1
fi

# Config path
export AUDIO_JEPA_CONFIG="${LOG_FOLDER}/.hydra/config.yaml"
if [ ! -f "$AUDIO_JEPA_CONFIG" ]; then
    echo "Error: Config file not found at $AUDIO_JEPA_CONFIG"
    exit 1
fi

# Checkpoint path
# Find the last checkpoint (last.ckpt)
CHECKPOINT_PATH=$(find "${LOG_FOLDER}/checkpoints" -name "last.ckpt" -print -quit)

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: No checkpoint matching 'last.ckpt' found in ${LOG_FOLDER}/checkpoints"
    exit 1
fi

export AUDIO_JEPA_CHECKPOINT="$CHECKPOINT_PATH"

echo "Using Config: $AUDIO_JEPA_CONFIG"
echo "Using Checkpoint: $AUDIO_JEPA_CHECKPOINT"

# Common exports
# export PYTHONWARNINGS="ignore::DeprecationWarning"
