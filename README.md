# A Structural Model for Contextual Code Changes

This repository contains the PyTorch implementation of the neural network _**C3PO**_, along with all the required code and data to reproduce our results.
Our code can be easily extended to other programming languages since the PyTorch network is agnostic to the input programming language (see [Extending to other languages](#Extending-to-other-languages)).
We also provide a with C# extractor for preprocessing the (raw) input code and explain how to implement such an extractor for other input programming languages.


Table of Contents
=================
  * [Requirements](#Requirements)
  * [Reproducing the paper results](#Reproducing-the-paper-results)
  * [Training new models](#Training-new-models)
  * [Configuration](#Configuration)
  * [Extending to other languages](#Extending-to-other-languages)

## Requirements
  * [python3](https://www.linuxbabe.com/ubuntu/install-python-3-6-ubuntu-16-04-16-10-17-04) 
  * For training and evaluating _**C3PO**_ and LaserTagger: PyTorch 1.4 or newer ([install](https://pytorch.org/get-started/locally/)). 
  To check PyTorch version:
    ```
    python3 -c 'import torch; print(torch.__version__)'
    ```
The following libraries can be installed by calling
```
    pip install -r requirements.txt
```
 * For training SequenceR and Path2Tree baselines: [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py#requirements). We provide here the necessary command lines to train these models with OpenNMT-py, but we also provide the raw data that can be used in any sequence-to-sequence framework. We also provide the prediction logs and the script that computes the accuracy from the logs, so installing OpenNMT-py is not required.
 * [tqdm](https://pypi.org/project/tqdm/)
 * [gitpython](https://gitpython.readthedocs.io/en/stable/intro.html)
 * [pygments](https://pypi.org/project/Pygments/)
 * [biopython](https://biopython.org/wiki/Download)
 * [scipy](https://pypi.org/project/scipy/)

### Hardware
A CPU is sufficient for validating the results of our trained models. A GPU is only required to train models from scratch.


## Reproducing the paper results
Here we provide the instructions for reproducing the main results from the paper. More detailed instructions of how to re-use our framework for training new models and new datasets is provided in the next sections.

Running the script `scripts/reproduce_table_2.sh` will reproduce the results reported in Table 2.
```
source scripts/reproduce_table_2.sh
```

Running the script `scripts/reproduce_table_3.sh` will reproduce the results reported in Table 3.
```
source scripts/reproduce_table_3.sh
```

### Reproducing _**C3PO**_ result using a pre-trained model
To perform inference on the test set using our pre-trained model, run the script `scripts/reproduce_C3PO_pre_trained.sh` (takes approximatly 30 minutes on a laptop without a GPU).
```
source scripts/reproduce_C3PO_pre_trained.sh
```

## Creating the dataset from scratch
Here we provide the instructions for creating a **new** dataset (other than the dataset experimented with in the paper), by cloning repositories from GitHub and processing them. 

#### Note: The process of creating new datasets can take a few days.

Please follow these steps:
1) Fill the `DataCreation/sampled_repos.txt` with all the desired repositories. Each lined corresponde to one repo - `<repo_name>\t<repo_url>`.
2) Fill the `splits_50.json` with the train-val-test splits.
3) Run the `scripts/fetch_repos.sh` to fetch these repos to your machine.
    ```
    source scripts/fetch_repos.sh
    ```
4) Run the `scripts/extract_commits.sh` to extract commits from these repos.
    ```
    source scripts/extract_commits.sh
    ```
5) Run the `scripts/create_commits_map.sh` to extract changes from the commits.
    ```
    source scripts/create_commits_map.sh
    ```
6) Run the `scripts/create_dataset.sh` to create the data for all the models (ours and the baselines).
    ```
    source scripts/create_dataset.sh
    ```
7) Run the `scripts/preprocess_c3po.sh` to preprocess the data for our model (_**C3PO**_).
    ```
    source scripts/preprocess_c3po.sh
    ```
8) Run the `scripts/preprocess_baselines.sh` to preprocess the data for the baselines.
    ```
    source scripts/preprocess_baselines.sh
    ```

Now you can find the following datasets and train new models as desrcibed in the next sections:
* `dataset_50_new` - _**C3PO**_
* `dataset_50_NMT` - SequenceR
* `dataset_50_path2tree` - Path2Tree
* `dataset_50_laser` - LaserTagger


[Here](https://drive.google.com/drive/folders/1x0BvD6T4q4Z8oym4MFJZSvYwkUb8XDAD?usp=sharing) you can find our original outputs of the final steps:
* `samples_50.tar.gz` - the output of step **6**.
* `dataset_50_new.tar.gz` - the output of step **7**.
* `dataset_50_baselines.zip` - the output of step **8**.

You can download them, and use them for the next sections.
Please extract the archives as follows:
* `samples_50.tar.gz` in `DataCreation`.
* `dataset_50_new.tar.gz` in the root folder.
* `dataset_50_baselines.zip` in the root folder.

## Training new models
After applying the step in the last section, you can use the created datasets to train new models.

### Training _**C3PO**_ model 
##### Training
Running the script `scripts/train_c3po.sh` will train a new model which will be located in `checkpoints/50_new_exp.pt`. Note that you can change the hyper-parameters (see [Configuration](#Configuration))
```
source scripts/train_c3po.sh
```
##### Evaluating
In order to test the trained model, run
```
source scripts/test_c3po.sh
```
### Training baselines models
For the following models, we used the OpenNMT-py framework:
* SequenceR LSTM
* SequenceR Transformer
* Path2Tree LSTM
* Path2Tree Transformer

Details about all flags that we used (which can be found in the following scripts) can be found at the [OpenNMT-py documentation](https://opennmt.net/OpenNMT-py/options/preprocess.html). Additional information about training a Transformer can be found in the [OpenNMT-py FAQ](https://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-the-transformer-model).

#### SequenceR LSTM
##### Training
Running the script `scripts/train_sequencer_lstm.sh` will train a new model which will be located in `dataset_50_NMT/changes/model_lstm`. Note that you need to stop the training manualy.
```
source scripts/train_sequencer_lstm.sh
```
##### Evaluating
After the training, you can test your model by first editing `scripts/test_sequencer_lstm.sh` and chaning the value of the variable `MODEL` to the desirable trained model and then running the edited script.
```
source scripts/test_sequencer_lstm.sh
```
#### SequenceR Transformer
##### Training
Running the script `scripts/train_sequencer_transformer.sh` will train a new model which will be located in `dataset_50_NMT/changes/model_transformer`. Note that you need to stop the training manualy. 
```
source scripts/train_sequencer_transformer.sh
```
##### Evaluating
After the training, you can test your model by first editing `scripts/test_sequencer_transformer.sh` and chaning the value of the variable `MODEL` to the desirable trained model and then running the edited script.
```
source scripts/test_sequencer_transformer.sh
```
#### Path2Tree LSTM
##### Training
Running the script `scripts/train_path2tree_lstm.sh` will train a new model which will be located in `dataset_50_path2tree/model_lstm`. Note that you need to stop the training manualy.
```
source scripts/train_path2tree_lstm.sh
```
##### Evaluating
After the training, you can test your model by first editing `scripts/test_path2tree_lstm.sh` and chaning the value of the variable `MODEL` to the desirable trained model and then running the edited script.
```
source scripts/test_path2tree_lstm.sh
```
#### Path2Tree Transformer
##### Training
Running the script `scripts/train_path2tree_transformer.sh` will train a new model which will be located in `dataset_50_path2tree/model_transformer`. Note that you need to stop the training manualy.
```
source scripts/train_path2tree_transformer.sh
```
##### Evaluating
After the training, you can test your model by first editing `scripts/test_path2tree_transformer.sh` and chaning the value of the variable `MODEL` to the desirable trained model and then running the edited script.
```
source scripts/test_path2tree_transformer.sh
```
#### LaserTagger LSTM
Running the script `scripts/train_lasertagger_lstm.sh` will train and evaluate a new model which will be located in `LaserTagger/checkpoints/model_lstm.pt`.
```
source scripts/train_lasertagger_lstm.sh
```
#### LaserTagger Transformer
Running the script `scripts/train_lasertagger_transformer.sh` will train and evaluate a new model which will be located in `LaserTagger/checkpoints/model_transformer.pt`.
```
source scripts/train_lasertagger_transformer.sh
```
### Training ablation models
#### Training _**C3PO**_ - no context model 
##### Training
Running the script `scripts/train_c3po_no_context.sh` will train a new model which will be located in `checkpoints/50_no_ctx.pt`.
```
source scripts/train_c3po_no_context.sh
```
##### Evaluating
In order to test the trained model, run
```
source scripts/test_c3po_no_context.sh
```
#### Training _**C3PO**_ - textual context model 
##### Training
Running the script `scripts/train_c3po_txt_context.sh` will train a new model which will be located in `checkpoints/50_txt_ctx.pt`.
```
source scripts/train_c3po_txt_context.sh
```
##### Evaluating
In order to test the trained model, run
```
source scripts/test_c3po_txt_context.sh
```
#### LaserTagger Transformer - no context model
Running the script `scripts/train_lasertagger_no_context.sh` will train and evaluate a new model which will be located in `LaserTagger/checkpoints/50_exp_transformer_no_ctx.pt`.
```
source scripts/train_lasertagger_no_context.sh
```
#### LaserTagger Transformer - path-based context model
Running the script `scripts/train_lasertagger_path_context.sh` will train and evaluate a new model which will be located in `LaserTagger/checkpoints/50_exp_transformer_path_ctx.pt`.
```
source scripts/train_lasertagger_path_context.sh
```


## Configuration
Changing _**C3PO**_ hyper-parameters is possible by editing the file `C3PO/config.py`
Here are some of them:

#### input_dim = 64
Embedding size of subtokens and node values.
#### hidden_dim = 128
Hidden states size.
#### num_of_layers = 1
LSTM's number of layers.
#### max_seq_len = 20
The max number of operation that can be predicted in test time.
#### dropout = 0.25
Dropout value.
#### early_stopping = 10
Controlling early stopping: how many epochs of no improvement should training continue before stopping.
#### batch_size = 64
Batch size during training.
#### lr = 0.001
Learning rate during training.
#### optim = 'adam'
The optimizer used during training.


## Extending to other languages
Since our model is agnostic to the input programming language, one can use it with other languages. To do that, the preprocessing script `preprocessing.py` needs to get the following:
Projects dir that contains directories where each one of them corespondes to one git project.
Each git project dir need to contain to following files:
* `<project_name>.path` - the AST paths of the samples.
* `<project_name>.path_op` - the associated operation for each path.
* `<project_name>.label` - the true label of the sample.
* `<project_name>.before_ctx_path` - the paths that represent changes in the preceding  context.
* `<project_name>.after_ctx_path` - the paths that represent changes in the succeeding context.
* `<project_name>.before_ctx_filtered` - the normalized textual representation (with interated changes) of the preceding  context.
* `<project_name>.after_ctx_filtered` - the normalized textual representation (with interated changes) of the succeeding context.

To train the other baselines, these project dirs also have to contain these files:
* `<project_name>.integrated_change_filtered` - integrated changes in the textual representation of the samples.
* `<project_name>.ids` - the id-value of each node in the AST.
* `<project_name>.edit_script` - human readable edit script for each sample.
* `<project_name>.before_filtered` - the textual representation of the code before the change.
* `<project_name>.before_normalized_filtered` - the noramzlied textual representation of the code before the change.
* `<project_name>.after_filtered` - the textual representation of the code after the change.
* `<project_name>.after_normalized_filtered` - the noramzlied textual representation of the code after the change.
* `<project_name>.before_ctx_before_filtered` - the textual representation of the preceding  context before its changes.
* `<project_name>.before_ctx_before_noramlized_filtered` - the normalized textual representation of the preceding  context before its changes.
* `<project_name>.before_ctx_after_filtered` - the textual representation of the preceding context after its changes.
* `<project_name>.before_ctx_after_normalized_filtered` - the normalized textual representation of the preceding context after its changes.
* `<project_name>.after_ctx_before_filtered` - the textual representation of the succeeding context before its changes.
* `<project_name>.after_ctx_before_noramlized_filtered` - the normalized textual representation of the succeeding context before its changes.
* `<project_name>.after_ctx_after_filtered` - the textual representation of the succeeding context after its changes.
* `<project_name>.after_ctx_after_normalized_filtered` - the normalized textual representation of the succeeding context after its changes.

Note that these files should be *line-aligned*.
In all these files, each sample corresponds to a specific line. I.e, the first line in all the files above, corresponds to the same sample.
You can find an example of these files in `samples_50_example`.

Lastly, a file that contains the train-val-test splits need to be provided. You can see an example - `splits_50.json`.

## Citations
If you want to cite this work, please use this bibtex entry:
```

@article{10.1145/3428283,
author = {Brody, Shaked and Alon, Uri and Yahav, Eran},
title = {A Structural Model for Contextual Code Changes},
year = {2020},
issue_date = {November 2020},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {4},
number = {OOPSLA},
url = {https://doi.org/10.1145/3428283},
doi = {10.1145/3428283},
abstract = {We address the problem of predicting edit completions based on a learned model that was trained on past edits. Given a code snippet that is partially edited, our goal is to predict a completion of the edit for the rest of the snippet. We refer to this task as the EditCompletion task and present a novel approach for tackling it. The main idea is to directly represent structural edits. This allows us to model the likelihood of the edit itself, rather than learning the likelihood of the edited code. We represent an edit operation as a path in the program’s Abstract Syntax Tree (AST), originating from the source of the edit to the target of the edit. Using this representation, we present a powerful and lightweight neural model for the EditCompletion task. We conduct a thorough evaluation, comparing our approach to a variety of representation and modeling approaches that are driven by multiple strong models such as LSTMs, Transformers, and neural CRFs. Our experiments show that our model achieves a 28% relative gain over state-of-the-art sequential models and 2\texttimes{} higher accuracy than syntactic models that learn to generate the edited code, as opposed to modeling the edits directly. Our code, dataset, and trained models are publicly available at https://github.com/tech-srl/c3po/ .},
journal = {Proc. ACM Program. Lang.},
month = nov,
articleno = {215},
numpages = {28},
keywords = {Machine Learning, Neural Models of Code, Edit Completions}
}
```
