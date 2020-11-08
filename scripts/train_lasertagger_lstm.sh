#!/usr/bin/env bash
PYTHON=python
LASER_OUT_DIR="dataset_50_laser"
SCRIPT="LaserTagger/main.py"

${PYTHON} ${SCRIPT} --expname "50_exp_lstm" --data ${LASER_OUT_DIR} --backbone "lstm"