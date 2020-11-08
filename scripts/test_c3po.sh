#!/usr/bin/env bash
PYTHON=python
OUT_DIR="data_50_new/"
CHECKPOINT="C3PO/checkpoints/50_new_exp.pt"

${PYTHON} C3PO/main.py --expname "50_new_exp_test" --data ${OUT_DIR} --load_checkpoint ${CHECKPOINT} --inference "true"