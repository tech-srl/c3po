#!/usr/bin/env bash
PYTHON=python
LASER_OUT_DIR="dataset_50_laser"
SCRIPT="LaserTagger/main.py"
CHECKPOINT="LaserTagger/checkpoints/50_exp_transformer_path_ctx.pt"

${PYTHON} ${SCRIPT} --expname "50_exp_transformer_path_ctx_test" --data ${LASER_OUT_DIR} --backbone 'transformer' --load_checkpoint ${CHECKPOINT} --inference "true" --num_of_layers 4 --context_mode "path"