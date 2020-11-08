#!/usr/bin/env bash

PYTHON=python
SCRIPT="DataCreation/create_dataset.py"
COMMITS_DIR="DataCreation/commit_data_processed"
OUT_DIR="DataCreation/samples_50"

${PYTHON} ${SCRIPT} ${COMMITS_DIR} ${OUT_DIR}

JAR="DataCreation/Extractor/target/Extractor-1.0-SNAPSHOT.jar"
MAX_NODES=50

# This file consists of the observed parent-child relations of the ASTs in the train split
LEGAL_CHILDREN_MAP="DataCreation/legalChildrenMap.ser"

cp ${LEGAL_CHILDREN_MAP} ${OUT_DIR}

java -cp ${JAR} Extractor.App --projects_dir ${OUT_DIR} --max_nodes ${MAX_NODES}

NMT_SCRIPT="DataCreation/create_NMT_splits.py"
SPLITS="splits_50.json"
NMT_OUT_DIR="dataset_50_NMT"

${PYTHON} ${NMT_SCRIPT} ${OUT_DIR} ${SPLITS} ${NMT_OUT_DIR}

PATH2TREE_SCRIPT="DataCreation/create_path2tree_splits.py"
PATH2TREE_OUT_DIR="dataset_50_path2tree"

${PYTHON} ${PATH2TREE_SCRIPT} ${OUT_DIR} ${SPLITS} ${PATH2TREE_OUT_DIR}

LASER_SCRIPT="DataCreation/LaserTagger/create_LaserTagger_splits.py"
LASER_OUT_DIR="dataset_50_laser"

${PYTHON} ${LASER_SCRIPT} ${OUT_DIR} ${SPLITS} ${LASER_OUT_DIR}