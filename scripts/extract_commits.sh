#!/usr/bin/env bash

PYTHON=python
EXTRACT_COMMITS="DataCreation/extract_commits.py"
REPOS="DataCreation/GitRepos"
OUT_DIR="DataCreation/commit_data"

${PYTHON} ${EXTRACT_COMMITS} ${OUT_DIR} ${REPOS}