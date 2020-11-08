#!/usr/bin/env bash

PYTHON=python
SCRIPT="C3PO/preprocess.py"
OUT_DIR="data_50_new"
PROJECTS_DIR="DataCreation/samples_50"
SPLITS="splits_50.json"
MIN_APPEARANCE=2
VOCAB_SIZE=520

${PYTHON} ${SCRIPT} --out_dir ${OUT_DIR} --projects_dir ${PROJECTS_DIR} --split_path ${SPLITS} --min_appearance ${MIN_APPEARANCE} --max_vocab_size ${VOCAB_SIZE}
