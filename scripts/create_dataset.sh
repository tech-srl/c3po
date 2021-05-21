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

SPLITS="splits_50.json"

########################################################################
# In case you want to create it from scratch, uncomment the lines bellow
########################################################################
# SCRIPT_B="DataCreation/copy_train_projects.py"
# TRAIN_OUT_DIR="DataCreation/train_samples_50"
# ${PYTHON} ${SCRIPT_B} ${OUT_DIR} ${TRAIN_OUT_DIR} ${SPLITS}
# java -cp ${JAR} Extractor.App --projects_dir ${TRAIN_OUT_DIR} --max_nodes ${MAX_NODES}
# LEGAL_CHILDREN_MAP="${TRAIN_OUT_DIR}/legalChildrenMap.ser"
#########################################################################

cp ${LEGAL_CHILDREN_MAP} ${OUT_DIR}

java -cp ${JAR} Extractor.App --projects_dir ${OUT_DIR} --max_nodes ${MAX_NODES}

NMT_SCRIPT="DataCreation/create_NMT_splits.py"

NMT_OUT_DIR="dataset_50_NMT"

${PYTHON} ${NMT_SCRIPT} ${OUT_DIR} ${SPLITS} ${NMT_OUT_DIR}

PATH2TREE_SCRIPT="DataCreation/create_path2tree_splits.py"
PATH2TREE_OUT_DIR="dataset_50_path2tree"

${PYTHON} ${PATH2TREE_SCRIPT} ${OUT_DIR} ${SPLITS} ${PATH2TREE_OUT_DIR}

LASER_SCRIPT="DataCreation/LaserTagger/create_LaserTagger_splits.py"
LASER_OUT_DIR="dataset_50_laser"

${PYTHON} ${LASER_SCRIPT} ${OUT_DIR} ${SPLITS} ${LASER_OUT_DIR}
