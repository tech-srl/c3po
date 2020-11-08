#!/usr/bin/env bash
PYTHON=python
OUT_DIR="data_50_new/"
CHECKPOINT="C3PO/checkpoints/50_txt_ctx.pt"

${PYTHON} main.py --expname "50_txt_ctx__test" --data ${OUT_DIR} --load_checkpoint ${CHECKPOINT} --inference "true" --context_mode "txt"