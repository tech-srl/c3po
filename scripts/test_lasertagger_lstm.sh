#!/usr/bin/env bash
PYTHON=python
LASER_OUT_DIR="dataset_50_laser"
SCRIPT="LaserTagger/main.py"
CHECKPOINT="LaserTagger/checkpoints/50_exp_lstm.pt"

${PYTHON} ${SCRIPT} --expname "50_exp_lstm_test" --data ${LASER_OUT_DIR} --backbone 'lstm' --load_checkpoint ${CHECKPOINT} --inference "true"