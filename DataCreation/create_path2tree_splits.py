import os
import multiprocessing as mp
from tqdm import tqdm
import sys
import json
import re

PATH_DELIMITER = ' <~> '
BOC_WORD = '<@>'
EOC_WORD = '</@>'
DEL_WORD = '<DEL>'
DOWN_WORD = '<D>'
UP_WORD = '<U>'


def process_paths(paths_str):
    if paths_str.strip() == '':
        return ''
    paths = paths_str.split('\t')
    paths_formated = list()
    for path in paths:
        tokens = [t.strip() for t in path.split(" ") if t.strip() != DOWN_WORD and t.strip() != UP_WORD]
        positions = tokens[1::2]
        tokens = tokens[0::2]
        src = tokens[0].split("|")
        src = "( " + " , ".join(src) + " )"
        tgt = tokens[-1].split("|")
        tgt = "( " + " , ".join(tgt) + " )"
        tokens = tokens[1:-1]
        tokens = [src] + tokens + [tgt]
        formated_path = " ".join(map(lambda t: t[0] + " " + t[1], zip(tokens, positions)))
        paths_formated.append(formated_path)
    return PATH_DELIMITER.join(paths_formated)


def processes_samples_task(args):
    project_id, dir_path = args
    if not os.path.isdir(dir_path):
        return None, None
    with open(os.path.join(dir_path, project_id + ".before_ctx_path"), "r") as f:
        before_ctx = f.readlines()
    with open(os.path.join(dir_path, project_id + ".after_ctx_path"), "r") as f:
        after_ctx = f.readlines()
    with open(os.path.join(dir_path, project_id + ".before_ast_trees"), "r") as f:
        before = f.readlines()
    with open(os.path.join(dir_path, project_id + ".after_ast_trees"), "r") as f:
        after = f.readlines()
    res = list()
    for i in range(len(before)):
        before_ = before[i].strip()
        if before_ == '':
            continue
        after_ = after[i].strip()
        before_paths = process_paths(before_ctx[i])
        after_paths = process_paths(after_ctx[i])
        source = before_paths.strip() + " " + BOC_WORD + " " + before_ + " " + EOC_WORD + " " + after_paths.strip()
        target = after_
        res.append((source, target))
    return res, project_id


def tokenize_for_training(samples_path, out_path, split_json):
    with open(split_json, "r") as f:
        splits = json.load(f)
    train = splits['train']
    val = splits['dev']
    test = splits['test']
    train_samples, val_samples, test_samples = list(), list(), list()
    train_projects, val_projects, test_projects = list(), list(), list()
    print("Listing files...", file=sys.stderr)
    files = list(map(lambda p: (p, os.path.join(samples_path, p)), os.listdir(samples_path)))
    print("Processing...", file=sys.stderr)
    projects_counts = dict()
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        with tqdm(total=len(files)) as pbar:
            for samples_list, project_id in pool.imap(processes_samples_task, files):
                pbar.update()
                if project_id in train:
                    train_samples += samples_list
                    train_projects += len(samples_list) * [project_id]
                    projects_counts[project_id] = (len(samples_list), "Train")
                elif project_id in val:
                    val_samples += samples_list
                    val_projects += len(samples_list) * [project_id]
                    projects_counts[project_id] = (len(samples_list), "Val")
                elif project_id in test:
                    test_samples += samples_list
                    test_projects += len(samples_list) * [project_id]
                    projects_counts[project_id] = (len(samples_list), "Test")
    # [print("project id: {} num_of_samples: {} split: {}".format(k, v[0], v[1])) for k, v in projects_counts.items()]
    print("Train: {}, Val: {}, Test: {}".format(len(train_samples), len(val_samples), len(test_samples)))
    train_src, train_dst = zip(*train_samples)
    val_src, val_dst = zip(*val_samples)
    test_src, test_dst = zip(*test_samples)

    print("Writing data...", file=sys.stderr)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    with open(os.path.join(out_path, "train.src"), "w") as f:
        f.write("\n".join(train_src))
    with open(os.path.join(out_path, "val.src"), "w") as f:
        f.write("\n".join(val_src))
    with open(os.path.join(out_path, "test.src"), "w") as f:
        f.write("\n".join(test_src))

    with open(os.path.join(out_path, "train.dst"), "w") as f:
        f.write("\n".join(train_dst))
    with open(os.path.join(out_path, "val.dst"), "w") as f:
        f.write("\n".join(val_dst))
    with open(os.path.join(out_path, "test.dst"), "w") as f:
        f.write("\n".join(test_dst))

    with open(os.path.join(out_path, "train.projects"), "w") as f:
        f.write("\n".join(train_projects))
    with open(os.path.join(out_path, "val.projects"), "w") as f:
        f.write("\n".join(val_projects))
    with open(os.path.join(out_path, "test.projects"), "w") as f:
        f.write("\n".join(test_projects))

    os.mkdir(os.path.join(out_path, "model_lstm"))
    os.mkdir(os.path.join(out_path, "results_lstm"))
    os.mkdir(os.path.join(out_path, "model_transformer"))
    os.mkdir(os.path.join(out_path, "results_transformer"))
    print("Done!", file=sys.stderr)


if __name__ == '__main__':
    samples_path = os.path.abspath(sys.argv[1])
    split_json = os.path.abspath(sys.argv[2])
    out_path = os.path.abspath(sys.argv[3])
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    tokenize_for_training(samples_path, out_path, split_json)
