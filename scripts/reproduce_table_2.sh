#!/usr/bin/env bash

PYTHON=python
SEQUENCER="paperResults/sequencer"
PATH2TREE="paperResults/path2tree"
CREATE_TABLE="paperResults/create_table.py"
LASER_SCRIPT="paperResults/print_results.py"
LASER_LSTM="paperResults/lasertagger/lstm/test_50_test_DEBUG.log"
LASER_TRANSFORMER="paperResults/lasertagger/transformer/test_50_transformer.log"

echo 'Table 2:'
echo 'Sequencer LSTM:'
${PYTHON} ${CREATE_TABLE} ${SEQUENCER} "lstm"

echo 'Sequencer Transformer:'
${PYTHON} ${CREATE_TABLE} ${SEQUENCER} "transformer"

echo 'LaserTagger LSTM:'
${PYTHON} ${LASER_SCRIPT} ${LASER_LSTM} "true"

echo 'LaserTagger Transformer:'
${PYTHON} ${LASER_SCRIPT} ${LASER_TRANSFORMER} "true"

echo 'Path2Tree LSTM:'
${PYTHON} ${CREATE_TABLE} ${PATH2TREE} "lstm"

echo 'Path2Tree Transformer:'
${PYTHON} ${CREATE_TABLE} ${PATH2TREE} "transformer"
echo 'Note: these results were reported in the paper as lower by 0.1 due to an incorrect decimal value rounding (Table 2). We will fix that in the camera ready version.'


C3PO_SCRIPT="paperResults/print_results.py"
C3PO="paperResults/c3po/test_50_lstm_2.log"

echo 'C3PO:'
${PYTHON} ${C3PO_SCRIPT} ${C3PO} "false"

