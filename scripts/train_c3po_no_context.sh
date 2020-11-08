PYTHON=python
SCRIPT="C3PO/main.py"
OUT_DIR="data_50_new/"

${PYTHON} ${SCRIPT} --expname "50_no_ctx" --data ${OUT_DIR} --context_mode "none"