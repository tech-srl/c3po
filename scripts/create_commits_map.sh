#!/usr/bin/env bash

PYTHON=python
SCRIPT="DataCreation/create_commits_map.py"
COMMITS_DIR="DataCreation/commit_data"
PROCESSED_COMMITS_DIR="DataCreation/commit_data_processed"

${PYTHON} ${SCRIPT} ${COMMITS_DIR} ${PROCESSED_COMMITS_DIR}
