#!/usr/bin/env bash

PYTHON=python
C3PO_SCRIPT="paperResults/print_results.py"
C3PO="paperResults/c3po/test_50_lstm_2.log"
LASER_SCRIPT="paperResults/print_results.py"
C3PO_NO_CTX="paperResults/c3po/test_50_no_ctx_test_DEBUG.log"
C3PO_TXT_CTX="paperResults/c3po/test_50_txt_ctx_test_DEBUG.log"
LASER_TRANSFORMER="paperResults/lasertagger/transformer/test_50_transformer.log"
LASER_TRANSFORMER_NO_CTX="paperResults/lasertagger/transformer/test_50_no_ctx_test_DEBUG.log"
LASER_TRANSFORMER_PATH_CTX="paperResults/lasertagger/transformer/test_50_path_ctx_test_DEBUG.log"


echo 'Table 3:'

echo 'LaserTagger - No context:'
${PYTHON} ${C3PO_SCRIPT} ${LASER_TRANSFORMER_NO_CTX} "true"
echo 'Note: these results were reported in the paper as lower by 0.1 due to an incorrect decimal value rounding (Table 3). We will fix that in the camera ready version.'

echo 'LaserTagger - Textual context:'
${PYTHON} ${C3PO_SCRIPT} ${LASER_TRANSFORMER} "true"

echo 'LaserTagger - Path based context:'
${PYTHON} ${C3PO_SCRIPT} ${LASER_TRANSFORMER_PATH_CTX} "true"

echo 'C3PO - No context:'
${PYTHON} ${C3PO_SCRIPT} ${C3PO_NO_CTX} "false"

echo 'C3PO - Textual context:'
${PYTHON} ${C3PO_SCRIPT} ${C3PO_TXT_CTX} "false"

echo 'C3PO - Path based context:'
${PYTHON} ${C3PO_SCRIPT} ${C3PO} "false"
