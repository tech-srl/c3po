PATH2TREE_OUT_DIR="dataset_50_path2tree"

P=${PATH2TREE_OUT_DIR}; onmt_train -data ${P}/data -save_model ${P}/model_lstm/model -encoder_type brnn -word_vec_size 512 -rnn_size 512 -layers 2 -global_attention general -train_steps 900000 -valid_steps 1000 -save_checkpoint_steps 1000 -optim adam  --learning_rate 0.001 --copy_attn -batch_size 256 --log_file ${P}/model_lstm/log.txt