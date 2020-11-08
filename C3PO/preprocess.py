import argparse
import os
from vocab import Vocab
import utils, Constants
import torch
import multiprocessing as mp
import sys
import traceback
from tqdm import tqdm
import json
from shutil import copyfile
import numpy as np
import re

def parse_args():
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--out_dir', default='data_50_new/',
                        help='output path')
    parser.add_argument('--projects_dir', default='DataCreation/samples_50_new',
                        help='path to preprocessed projects samples')
    parser.add_argument('--split_path', default='splits_50.json',
                        help='path to train-val-test split')
    parser.add_argument('--min_appearance', type=int, default=2,
                        help='remove all words that appear less then min_appearance in vocabulary')
    parser.add_argument('--max_vocab_size', type=int, default=520,
                        help='max vocabulary size')
    parser.add_argument('--ignore_directions', type=bool, default=True,
                        help='ignore <U>, <D> symbols in paths')

    args = parser.parse_args()
    return args


def create_file_list(dirs, suffix):
    if type(suffix) is str:
        suffix = [suffix]
    list_of_list = list()
    for s in suffix:
        list_of_list += [
            list(
                map(lambda x: os.path.join(dir, x),
                    filter(lambda x: x.endswith(s),
                           os.listdir(dir)
                           )))
            for dir in dirs
        ]
    return [item for sublist in list_of_list for item in sublist]


def create_vocabularies(train_dir, dev_dir, args):
    token_files_paths = create_file_list([train_dir, dev_dir], [".path", ".before_ctx_path", ".after_ctx_path"])
    token_files_ctx = create_file_list([train_dir, dev_dir], ".before_ctx_filtered") + create_file_list([train_dir, dev_dir], ".after_ctx_filtered")
    token_files_labels = create_file_list([train_dir, dev_dir], ".label")

    path_vocab_file = os.path.join(args.out_dir, 'path.vocab')
    path_vocab_object = os.path.join(args.out_dir, 'path.pth')

    src_tgt_vocab_file = os.path.join(args.out_dir, 'src_tgt.vocab')
    src_tgt_vocab_object = os.path.join(args.out_dir, 'src_tgt.pth')

    position_vocab_file = os.path.join(args.out_dir, 'position.vocab')
    position__vocab_object = os.path.join(args.out_dir, 'position.pth')

    utils.build_path_vocabs(token_files_paths, path_vocab_file, src_tgt_vocab_file, position_vocab_file, ignore_directions=args.ignore_directions, min_appearance=args.min_appearance, max_size=args.max_vocab_size)
    # get vocab object from vocab file previously written
    path_vocab = Vocab(filename=path_vocab_file,
                  data=[Constants.PAD_WORD, Constants.EOS_WORD, Constants.BOS_WORD, Constants.UNK_WORD]
                  )
    torch.save(path_vocab, path_vocab_object)
    src_tgt_vocab = Vocab(filename=src_tgt_vocab_file,
                  data=[Constants.PAD_WORD, Constants.EOS_WORD, Constants.BOS_WORD, Constants.UNK_WORD]
                  )
    torch.save(src_tgt_vocab, src_tgt_vocab_object)
    position_vocab = Vocab(filename=position_vocab_file,
                  data=[Constants.PAD_WORD, Constants.UNK_WORD]
                  )
    torch.save(position_vocab, position__vocab_object)
    print('path vocabulary size : {}'.format(path_vocab.size()))
    print('src_tgt vocabulary size : {}'.format(src_tgt_vocab.size()))
    print('position vocabulary size : {}'.format(position_vocab.size()))

    ctx_vocab_file = os.path.join(args.out_dir, 'ctx.vocab')
    ctx_vocab_object = os.path.join(args.out_dir, 'ctx.pth')
    utils.build_label_vocab(token_files_ctx, ctx_vocab_file, min_appearance=args.min_appearance, max_size=args.max_vocab_size)
    ctx_vocab = Vocab(filename=ctx_vocab_file,
                        data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    torch.save(ctx_vocab, ctx_vocab_object)
    print('context vocabulary size : {}'.format(ctx_vocab.size()))



    labels_vocab_file = os.path.join(args.out_dir, 'labels.vocab')
    labels_vocab_object = os.path.join(args.out_dir, 'labels.pth')
    utils.build_label_vocab(token_files_labels, labels_vocab_file, min_appearance=0)
    label_vocab = Vocab(data=[Constants.PAD_WORD, Constants.EOS_WORD])
    torch.save(label_vocab, labels_vocab_object)
    print('labels vocabulary size : {}'.format(label_vocab.size()))

    return path_vocab, src_tgt_vocab, position_vocab, ctx_vocab


def pad_sequences(seqs):
    seq_lengths = list(map(len, seqs))
    # assuming <PAD> index is 0 in vocabulary
    padded_seqs = np.zeros((len(seqs), max(seq_lengths)), dtype=int)
    for idx, (seq, seqlen) in enumerate(zip(seqs, seq_lengths)):
        padded_seqs[idx, :seqlen] = seq
    return padded_seqs, seq_lengths


def create_sample_from_project(args):
    source_dir, project_name, path_vocab, src_tgt_vocab, position_vocab, ctx_vocab, dest, ignore_directions = args
    res = []
    changes_pattern = re.compile(r"<%>.*?</%>")

    def extract_paths_aux(entry, prefix, paths):
        entry[prefix + 'paths'] = list()
        entry[prefix + 'paths_positions'] = list()
        entry[prefix + 'srcs'] = list()
        entry[prefix + 'srcs_positions'] = list()
        entry[prefix + 'tgts'] = list()
        entry[prefix + 'tgts_positions'] = list()
        for j, path in enumerate(paths):
            tokens_raw_with_directions = [t.strip() for t in path.split(" ")]
            tokens_raw = [t for t in tokens_raw_with_directions if t != Constants.DOWN_WORD and t != Constants.UP_WORD]
            positions = tokens_raw[1::2]
            tokens = tokens_raw[0::2]
            src = tokens[0]
            src_position = positions[0]
            tgt = tokens[-1]
            tgt_position = positions[-1]
            tokens = tokens[1:-1]
            positions = positions[1:-1]
            if ignore_directions is False:
                raise NotImplementedError
            entry[prefix + 'srcs'].append([src_tgt_vocab.getIndex(Constants.BOS_WORD)] + [src_tgt_vocab.getIndex(t.strip(), default=Constants.UNK_WORD) for t in src.split('|') if t != ''] + [src_tgt_vocab.getIndex(Constants.EOS_WORD)])
            entry[prefix + 'srcs_positions'].append(len(res[i][prefix + 'srcs'][-1]) * [position_vocab.getIndex(src_position.strip(), default="0")])
            entry[prefix + 'tgts'].append([src_tgt_vocab.getIndex(Constants.BOS_WORD)] + [src_tgt_vocab.getIndex(t.strip(), default=Constants.UNK_WORD) for t in tgt.split('|') if t != ''] + [src_tgt_vocab.getIndex(Constants.EOS_WORD)])
            entry[prefix + 'tgts_positions'].append(len(res[i][prefix + 'tgts'][-1]) * [position_vocab.getIndex(tgt_position.strip(), default="0")])
            entry[prefix + 'paths'].append([path_vocab.getIndex(Constants.BOS_WORD)] + [path_vocab.getIndex(t.strip(), default=Constants.UNK_WORD) for t in tokens] + [path_vocab.getIndex(Constants.EOS_WORD)])
            entry[prefix + 'paths_positions'].append([position_vocab.getIndex("0")] + [position_vocab.getIndex(t.strip(), default=Constants.UNK_WORD) for t in positions] + [position_vocab.getIndex("0")])

    try:
        with open(os.path.join(source_dir, project_name) + ".path", "r") as f:
            with open(os.path.join(source_dir, project_name) + ".label", "r") as fl:
                labels = fl.readlines()
            with open(os.path.join(source_dir, project_name) + ".path_op", "r") as fl:
                path_ops = fl.readlines()
            with open(os.path.join(source_dir, project_name) + ".before_ctx_path", "r") as fl:
                before_ctx_path = fl.readlines()
            with open(os.path.join(source_dir, project_name) + ".after_ctx_path", "r") as fl:
                after_ctx_path = fl.readlines()
            with open(os.path.join(source_dir, project_name) + ".before_ctx_filtered", "r") as fl:
                before_ctx = fl.readlines()
            with open(os.path.join(source_dir, project_name) + ".after_ctx_filtered", "r") as fl:
                after_ctx = fl.readlines()

            for i, line in enumerate(f.readlines()):
                if len(res) <= i:
                    res.append(dict())
                paths = line.split("\t")
                extract_paths_aux(res[i], '', paths)
                extract_paths_aux(res[i], 'before_ctx_', before_ctx_path[i].split("\t"))
                extract_paths_aux(res[i], 'after_ctx_', after_ctx_path[i].split("\t"))
                extract_paths_aux(res[i], 'none_before_ctx_', ' \n'.split("\t"))
                extract_paths_aux(res[i], 'none_after_ctx_', ' \n'.split("\t"))

                res[i]['paths_ops'] = path_ops[i].split('\t')
                cur_labels = labels[i].strip().split('\t')
                cur_labels = list(map(lambda s: s.split(' '), cur_labels))
                res[i]['label'] = list()
                for t in cur_labels:
                    if t[0] == Constants.MOV_WORD:
                        res[i]['label'].append(int(t[1]) + Constants.NUM_OF_CTRL_TOKENS)
                    elif t[0] == Constants.UPD_WORD:
                        res[i]['label'].append(int(t[1]) + len(paths) + Constants.NUM_OF_CTRL_TOKENS)
                    else:  # INS
                        res[i]['label'].append(int(t[1]) + 2 * len(paths) + Constants.NUM_OF_CTRL_TOKENS)
                res[i]['label'].append(Constants.EOS)
                res[i]['label_string'] = labels[i].rstrip('\n')
                res[i]['id'] = (project_name, i)
                before_ctx_changes = " ".join(map(lambda s: s.strip(), changes_pattern.findall(before_ctx[i])))
                after_ctx_changes = " ".join(map(lambda s: s.strip(), changes_pattern.findall(after_ctx[i])))
                changes = before_ctx_changes + " " + after_ctx_changes
                res[i]['ctx_txt'] = [ctx_vocab.getIndex(Constants.BOS_WORD)] + [ctx_vocab.getIndex(t.strip(), default=Constants.UNK_WORD) for t in [t.strip() for t in changes.split(" ")] if t != ''] + [ctx_vocab.getIndex(Constants.EOS_WORD)]
    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        traceback.print_exc(file=sys.stdout)
    for i, r in enumerate(res):
        torch.save(r, os.path.join(dest, project_name + "_{}.pth".format(i)))
    return project_name


def create_data_sets(source_dir, path_vocab, src_tgt_vocab, position_vocab, ctx_vocab, dest, args):
    ignore_directions = args.ignore_directions
    if not os.path.exists(dest):
        os.mkdir(dest)
    project_names = list(
                        map(lambda p: p.split(".path")[0],
                            filter(lambda x: x.endswith(".path"),
                                os.listdir(source_dir))))
    args = [(source_dir, project_name, path_vocab, src_tgt_vocab, position_vocab, ctx_vocab, dest, ignore_directions) for project_name in project_names]
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        with tqdm(total=len(project_names)) as pbar:
            for _ in pool.imap(create_sample_from_project, args):
                pbar.update()
    print("Total: {}".format(len(os.listdir(dest))))


def count_samples_in_dir(dir):
    count = 0
    project_names = set(map(lambda f: ".".join((f.split(".")[:-1])), os.listdir(dir)))
    for prject_name in project_names:
        befores = os.path.join(dir, prject_name + ".before_filtered")
        with open(befores, "r", encoding="utf8") as f:
            count += len(f.readlines())
    return count


def copy_samples(src_dirs, dst, override_samples):
    if os.path.exists(dst) and override_samples is False:
        return
    if not os.path.exists(dst):
        os.mkdir(dst)
    for dir in src_dirs:
        for file in os.listdir(dir):
            if file not in ["after.txt", "before.txt"]:
                copyfile(os.path.join(dir, file), os.path.join(dst, file))


def build_split(split_path, projects_path, out_dir, count_samples=False, override_samples=False):
    with open(split_path, "r") as f:
        splits = json.load(f)
    train_dirs = [os.path.join(projects_path, project) for project in splits['train']]
    dev_dirs = [os.path.join(projects_path, project) for project in splits['dev']]
    test_dirs = [os.path.join(projects_path, project) for project in splits['test']]

    train_dir = os.path.join(out_dir, "train_raw")
    dev_dir = os.path.join(out_dir, "dev_raw")
    test_dir = os.path.join(out_dir, "test_raw")

    copy_samples(train_dirs, train_dir, override_samples)
    copy_samples(dev_dirs, dev_dir, override_samples)
    copy_samples(test_dirs, test_dir, override_samples)

    if count_samples:
        print("Train: {} Dev: {} Test: {}".format(count_samples_in_dir(train_dir), count_samples_in_dir(dev_dir), count_samples_in_dir(test_dir)))
    return train_dir, dev_dir, test_dir


def create_stats_for_split(split_dir):
    res = {
        'DEL': 0,
        'MOV': 0,
        'UPD': 0,
        'INS': 0,
        'num_of_ops': 0,
        'num_of_paths': 0,
        'max_ctx_paths': 0,
        'max_ops': 0,
        'num_of_samples': 0,
        'DEL_portion': 0,
        'MOV_portion': 0,
        'UPD_portion': 0,
        'INS_portion': 0
    }
    edit_script_files = list(map(lambda f: os.path.join(split_dir, f), filter(lambda f: f.endswith('.edit_script'), os.listdir(split_dir))))
    for file in edit_script_files:
        with open(file, "r") as f:
            lines = f.readlines()
        for line in lines:
            res['num_of_samples'] += 1
            res['num_of_ops'] += len(line.split("\t"))
            res['max_ops'] = max(res['max_ops'], len(line.split("\t")))
            res['DEL'] += line.count("DEL")
            res['MOV'] += line.count("MOV")
            res['UPD'] += line.count("UPD")
            res['INS'] += line.count("INS")
    path_files = list(map(lambda f: os.path.join(split_dir, f), filter(lambda f: f.endswith('.path'), os.listdir(split_dir))))
    for file in path_files:
        with open(file, "r") as f:
            lines = f.readlines()
        for line in lines:
            res['num_of_paths'] += len(line.split("\t"))
    projects = set(map(lambda x: x.split(".path")[0], path_files))
    # counter = 0
    for p in projects:
        with open(p + ".before_ctx_path", "r") as f:
            before_lines = f.readlines()
        with open(p + ".after_ctx_path", "r") as f:
            after_lines = f.readlines()
        for i in range(len(before_lines)):
            total_paths = 0
            if before_lines[i].strip() != '':
                total_paths += len(before_lines[i].split("\t"))
            if after_lines[i].strip() != '':
                total_paths += len(after_lines[i].split("\t"))
            # if total_paths > 20:
            #     print(p, i, total_paths)
            #     counter += 1
            res['max_ctx_paths'] = max(total_paths, res['max_ctx_paths'])
    # print(counter)
    res['avg_ops'] = res['num_of_ops'] / res['num_of_samples']
    # res['DEL_portion'] = res['DEL'] / res['num_of_ops']
    # res['MOV_portion'] = res['MOV'] / res['num_of_ops']
    # res['UPD_portion'] = res['UPD'] / res['num_of_ops']
    # res['INS_portion'] = res['INS'] / res['num_of_ops']
    res['DEL_portion'] = (res['DEL'] / res['num_of_ops']) * res['avg_ops']
    res['MOV_portion'] = (res['MOV'] / res['num_of_ops']) * res['avg_ops']
    res['UPD_portion'] = (res['UPD'] / res['num_of_ops']) * res['avg_ops']
    res['INS_portion'] = (res['INS'] / res['num_of_ops']) * res['avg_ops']
    res['avg_paths'] = res['num_of_paths'] / res['num_of_samples']
    return res


def calc_statistics(train_dir, dev_dir, test_dir, out_dir):
    train_dict = create_stats_for_split(train_dir)
    dev_dict = create_stats_for_split(dev_dir)
    test_dict = create_stats_for_split(test_dir)
    template_str = "DEL: {:0.3f} MOV: {:0.3f} UPD: {:0.3f} INS: {:0.3f} OPS: {:0.3f} PATHS: {:0.3f} MAX_CTX_PATHS: {} MAX_OPS: {}"
    train_str = "Train:  " + template_str.format(train_dict['DEL_portion'], train_dict['MOV_portion'], train_dict['UPD_portion'], train_dict['INS_portion'], train_dict['avg_ops'], train_dict['avg_paths'], train_dict['max_ctx_paths'], train_dict['max_ops'])
    dev_str = "Dev:    " + template_str.format(dev_dict['DEL_portion'], dev_dict['MOV_portion'], dev_dict['UPD_portion'], dev_dict['INS_portion'], dev_dict['avg_ops'], dev_dict['avg_paths'], dev_dict['max_ctx_paths'], dev_dict['max_ops'])
    test_str = "Test:   " + template_str.format(test_dict['DEL_portion'], test_dict['MOV_portion'], test_dict['UPD_portion'], test_dict['INS_portion'], test_dict['avg_ops'], test_dict['avg_paths'], test_dict['max_ctx_paths'], test_dict['max_ops'])
    print(train_str)
    print(dev_str)
    print(test_str)
    all_num_of_ops = train_dict['num_of_ops'] + dev_dict['num_of_ops'] + test_dict['num_of_ops']
    all_num_of_paths = train_dict['num_of_paths'] + dev_dict['num_of_paths'] + test_dict['num_of_paths']
    all_samples = train_dict['num_of_samples'] + dev_dict['num_of_samples'] + test_dict['num_of_samples']
    all_DEL_portion = (train_dict['DEL'] + dev_dict['DEL'] + test_dict['DEL']) / all_num_of_ops
    all_MOV_portion = (train_dict['MOV'] + dev_dict['MOV'] + test_dict['MOV']) / all_num_of_ops
    all_UPD_portion = (train_dict['UPD'] + dev_dict['UPD'] + test_dict['UPD']) / all_num_of_ops
    all_INS_portion = (train_dict['INS'] + dev_dict['INS'] + test_dict['INS']) / all_num_of_ops
    all_str = "All:    DEL: {:0.3f} MOV: {:0.3f} UPD: {:0.3f} INS: {:0.3f} OPS: {:0.3f} PATHS: {:0.3f}".format(all_DEL_portion, all_MOV_portion, all_UPD_portion, all_INS_portion, all_num_of_ops / all_samples, all_num_of_paths / all_samples)
    print(all_str)
    with open(os.path.join(out_dir, 'stats.txt'), "w") as f:
        f.writelines([train_str + "\n", dev_str + "\n", test_str + "\n", all_str])


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    print("Copying examples")
    train_dir, dev_dir, test_dir = build_split(args.split_path, args.projects_dir, args.out_dir, count_samples=True)
    print("Creating statistics")
    calc_statistics(train_dir, dev_dir, test_dir, args.out_dir)
    print("Creating vocabulary")
    path_vocab, src_tgt_vocab, position_vocab, ctx_vocab = create_vocabularies(train_dir, dev_dir, args)
    print("Creating data pickles")
    print("Train")
    create_data_sets(train_dir, path_vocab, src_tgt_vocab, position_vocab, ctx_vocab, os.path.join(args.out_dir, 'train/'), args)
    print("Dev")
    create_data_sets(dev_dir, path_vocab, src_tgt_vocab, position_vocab, ctx_vocab, os.path.join(args.out_dir, 'dev/'), args)
    print("Test")
    create_data_sets(test_dir, path_vocab, src_tgt_vocab, position_vocab, ctx_vocab, os.path.join(args.out_dir, 'test/'), args)
    print("Done")