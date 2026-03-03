#!/bin/bash

# Usage: ./run_sweep.sh MODEL DATASET

MODEL="$1"
DATASET="$2"
DATA_DIR="data/${DATASET}"
if [ "$DATASET" == "codex-s" ] || [ "$DATASET" == "kinship" ] || [ "$DATASET" == "nations" ]; then
    walltime="12:00:00"
    cores=1
else
    walltime="48:00:00"
    cores=3
fi

if [ "$MODEL" == "ConvE" ]; then
    GPU_OPTION="--use_gpu"
    OAR_RESOURCES="gpu=1,walltime=$walltime"
else
    OAR_RESOURCES="core=$cores,walltime=$walltime"
fi

# Step 1: Init the sweep and capture the ID
echo "Init sweep for model=${MODEL}, dataset=${DATASET}..."
SWEEP_OUTPUT=$(python3 main.py --init_sweep --data_dir "$DATA_DIR" --model "$MODEL")

# Extract the ID from the output
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oE 'Create sweep with ID: ([a-z0-9]+)' | awk '{print $5}')

if [ -z "$SWEEP_ID" ]; then
  echo "Failed to retrieve sweep ID."
  exit 1
fi

echo "Sweep ID: $SWEEP_ID"

# Step 2: Submit 30 jobs OAR
for i in $(seq 1 30); do
  echo "Submit job $i/30"
  oarsub -l $OAR_RESOURCES -n sweep_${MODEL}_${DATASET}_${i} \
    "python3 main.py --sweep_id=$SWEEP_ID --data_dir $DATA_DIR --model $MODEL $GPU_OPTION"
done
