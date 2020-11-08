#!/usr/bin/env bash
PYTHON=python
LASER_OUT_DIR="dataset_50_laser"
SCRIPT="LaserTagger/main.py"

${PYTHON} ${SCRIPT} --expname "50_exp_transformer_no_ctx" --data ${LASER_OUT_DIR} --backbone "transformer" --num_of_layers 4 --context_mode "none"