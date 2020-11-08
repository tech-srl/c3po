#!/usr/bin/env bash

PYTHON=python
CRAWLER="DataCreation/crawl_gitgub.py"
REPO_LIST="DataCreation/sampled_repos.txt"
DEST="DataCreation/GitRepos"

${PYTHON} ${CRAWLER} ${REPO_LIST} ${DEST}
