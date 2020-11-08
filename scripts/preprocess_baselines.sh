#!/usr/bin/env bash


NMT_OUT_DIR="dataset_50_NMT/changes"
PATH2TREE_OUT_DIR="dataset_50_path2tree"

# OpenNMT
P=${NMT_OUT_DIR}; onmt_preprocess -train_src ${P}/train.src -train_tgt ${P}/train.dst -valid_src ${P}/val.src -valid_tgt ${P}/val.dst -save_data ${P}/data -src_seq_length_trunc 400 -src_seq_length 400 -tgt_seq_length_trunc 100 -tgt_seq_length 100 -src_vocab_size 520 -tgt_vocab_size 520 -dynamic_dict -share_vocab

P=${PATH2TREE_OUT_DIR}; onmt_preprocess -train_src ${P}/train.src -train_tgt ${P}/train.dst -valid_src ${P}/val.src -valid_tgt ${P}/val.dst -save_data ${P}/data -src_seq_length_trunc 400 -src_seq_length 400 -tgt_seq_length_trunc 100 -tgt_seq_length 100 -src_vocab_size 520 -tgt_vocab_size 520 -dynamic_dict -share_vocab

# LaserTagger
PYTHON=python
SCRIPT="LaserTagger/preprocess.py"
MIN_APPEARANCE=2
VOCAB_SIZE=520
LASER_OUT_DIR="dataset_50_laser"

${PYTHON} ${SCRIPT} --out_dir ${LASER_OUT_DIR} --min_appearance ${MIN_APPEARANCE} --max_vocab_size ${VOCAB_SIZE}
