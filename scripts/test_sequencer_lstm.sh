NMT_OUT_DIR="dataset_50_NMT/changes"
MODEL="model_lstm/model_step_36000.pt"

P=${NMT_OUT_DIR}; onmt_translate -model ${P}/${MODEL} -src ${P}/test.src -output ${P}/results_lstm/pred_test.txt -n_best 1 -beam_size 1 --replace_unk

PYTHON="python"
SCRIPT="paperResults/create_table.py"

${PYTHON} ${SCRIPT} ${NMT_OUT_DIR} "results_lstm"