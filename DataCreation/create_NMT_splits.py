import os
import multiprocessing as mp
from tqdm import tqdm
import sys
import json
import re
from create_path2tree_splits import process_paths

BOC_WORD = '<@>'
EOC_WORD = '</@>'
DEL_WORD = '<DEL>'


def process_sample(sample):
    before = " ".join(sample['before']).strip()
    if before.strip() == '':
        return None
    after = " ".join(sample['after']).strip()
    if after == '':
        after = DEL_WORD
    return(
        " ".join(sample['before_context']).strip() +
        " " + BOC_WORD + " " +
        before +
        " " + EOC_WORD + " " +
        " ".join(sample['after_context']).strip(),
        after,
        " ".join(sample['integrated_change']))


def processes_samples_task(args):
    project_id, dir_path, ctx_mode = args
    if not os.path.isdir(dir_path):
        return None, None
    with open(os.path.join(dir_path, project_id + ".before_ctx_filtered"), "r") as f:
        before_ctx = f.readlines()
    with open(os.path.join(dir_path, project_id + ".after_ctx_filtered"), "r") as f:
        after_ctx = f.readlines()
    with open(os.path.join(dir_path, project_id + ".before_ctx_before_normalized_filtered"), "r") as f:
        before_ctx_before = f.readlines()
    with open(os.path.join(dir_path, project_id + ".before_ctx_after_normalized_filtered"), "r") as f:
        before_ctx_after = f.readlines()
    with open(os.path.join(dir_path, project_id + ".after_ctx_before_normalized_filtered"), "r") as f:
        after_ctx_before = f.readlines()
    with open(os.path.join(dir_path, project_id + ".after_ctx_after_normalized_filtered"), "r") as f:
        after_ctx_after = f.readlines()
    with open(os.path.join(dir_path, project_id + ".before_normalized_filtered"), "r") as f:
        before = f.readlines()
    with open(os.path.join(dir_path, project_id + ".after_normalized_filtered"), "r") as f:
        after = f.readlines()
    with open(os.path.join(dir_path, project_id + ".before_ctx_path"), "r") as f:
        before_ctx_path = f.readlines()
    with open(os.path.join(dir_path, project_id + ".after_ctx_path"), "r") as f:
        after_ctx_path = f.readlines()

    changes_pattern = re.compile(r"<%>.*?</%>")

    res = list()
    for i in range(len(before)):
        before_ = before[i].strip()
        if before_ == '':
            continue
        after_ = after[i].strip()
        if after_ == '':
            after_ = DEL_WORD
        if ctx_mode == 'full':
            source = before_ctx[i].strip() + " " + BOC_WORD + " " + before_ + " " + EOC_WORD + " " + after_ctx[i].strip()
        elif ctx_mode == 'none':
            source = before_
        elif ctx_mode == 'before':
            source = before_ctx_before[i].strip() + " " + BOC_WORD + " " + before_ + " " + EOC_WORD + " " + after_ctx_before[i].strip()
        elif ctx_mode == 'after':
            source = before_ctx_after[i].strip() + " " + BOC_WORD + " " + before_ + " " + EOC_WORD + " " + after_ctx_after[i].strip()
        elif ctx_mode == 'changes':
            before_ctx_changes = " ".join(map(lambda s: s.strip(), changes_pattern.findall(before_ctx[i])))
            after_ctx_changes = " ".join(map(lambda s: s.strip(), changes_pattern.findall(after_ctx[i])))
            source = before_ctx_changes.strip() + " " + BOC_WORD + " " + before_ + " " + EOC_WORD + " " + after_ctx_changes.strip()
        elif ctx_mode == 'path':
            before_paths = process_paths(before_ctx_path[i])
            after_paths = process_paths(after_ctx_path[i])
            source = before_paths.strip() + " " + BOC_WORD + " " + before_ + " " + EOC_WORD + " " + after_paths.strip()
        else:
            raise Exception
        target = after_
        res.append((source, target))
    return res, project_id


def tokenize_for_training(samples_path, out_path, split_json, ctx_mode):
    with open(split_json, "r") as f:
        splits = json.load(f)
    train = splits['train']
    val = splits['dev']
    test = splits['test']
    train_samples, val_samples, test_samples = list(), list(), list()
    train_projects, val_projects, test_projects = list(), list(), list()
    print("Listing files...", file=sys.stderr)
    files = list(map(lambda p: (p, os.path.join(samples_path, p), ctx_mode), os.listdir(samples_path)))
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

    ctx_mode = 'changes'
    tokenize_for_training(samples_path, os.path.join(out_path, ctx_mode), split_json, ctx_mode)

