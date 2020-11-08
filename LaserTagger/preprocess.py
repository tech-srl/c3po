import argparse
import os
from vocab import Vocab
import Constants
import torch
from tqdm import tqdm

UP_WORD = '<U>'
DOWN_WORD = '<D>'
PATH_DELIMITER = ' <~> '

def parse_args():
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--out_dir', default='dataset_50_Laser/',
                        help='output path')
    parser.add_argument('--min_appearance', type=int, default=2,
                        help='remove all words that appear less then min_appearance in vocabulary')
    parser.add_argument('--max_vocab_size', type=int, default=520,
                        help='max vocabulary size')
    args = parser.parse_args()
    return args


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


def build_label_vocab(tokens_filenames, vocabfile=None, min_appearance=0, max_size=None, is_path=False):
    vocab = set()
    label_count = dict()
    if tokens_filenames is not None:
        for filename in tqdm(tokens_filenames):
            with open(filename, 'r') as f:
                for line in f:
                    if is_path:
                        line = process_paths(line.rstrip('\n'))
                    tokens = [t.strip() for t in line.rstrip('\n').split(" ")if t.strip() != DOWN_WORD and t.strip() != UP_WORD]
                    for token in tokens:
                        if token == '':
                            continue
                        label_count[token] = 1 if token not in label_count.keys() else label_count[token] + 1
                        if label_count[token] >= min_appearance:
                            vocab.add(token)
    if max_size is not None:
        sorted_vocab = [(count, label) for label, count in label_count.items()]
        sorted(sorted_vocab, key=lambda t: t[0], reverse=True)
        sorted_vocab = sorted_vocab[:max_size]
        _, sorted_vocab = zip(*sorted_vocab)
        vocab = set(sorted_vocab)
    if vocabfile is not None:
        with open(vocabfile, 'w') as f:
            for token in sorted(vocab):
                f.write(token + '\n')
    return vocab


def create_vocabularies(out_dir, args):
    src_file = os.path.join(out_dir, 'src.vocab')
    src_object = os.path.join(out_dir, 'src.pth')
    token_files_src = [os.path.join(out_dir, "train.src"), os.path.join(out_dir, "val.src")]
    build_label_vocab(token_files_src, src_file, min_appearance=args.min_appearance, max_size=args.max_vocab_size)
    src_vocab = Vocab(filename=src_file,
                      data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    torch.save(src_vocab, src_object)
    print('src vocabulary size : {}'.format(src_vocab.size()))
    tgt_file = os.path.join(out_dir, 'tgt.vocab')
    tgt_object = os.path.join(out_dir, 'tgt.pth')
    token_files_tgt = [os.path.join(out_dir, "train.dst"), os.path.join(out_dir, "val.dst")]
    build_label_vocab(token_files_tgt, tgt_file, min_appearance=args.min_appearance, max_size=args.max_vocab_size)
    tgt_vocab = Vocab(filename=tgt_file,
                      data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    torch.save(tgt_vocab, tgt_object)
    print('tgt vocabulary size : {}'.format(tgt_vocab.size()))

    ctx_vocab_file = os.path.join(out_dir, 'ctx.vocab')
    ctx_vocab_object = os.path.join(out_dir, 'ctx.pth')
    token_files_ctx = [os.path.join(out_dir, "train.before_ctx"), os.path.join(out_dir, "val.before_ctx")]
    build_label_vocab(token_files_ctx, ctx_vocab_file, min_appearance=args.min_appearance, max_size=args.max_vocab_size)
    ctx_vocab = Vocab(filename=ctx_vocab_file,
                        data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    torch.save(ctx_vocab, ctx_vocab_object)
    print('context vocabulary size : {}'.format(ctx_vocab.size()))

    ctx_path_vocab_file = os.path.join(out_dir, 'ctx_path.vocab')
    ctx_path_vocab_object = os.path.join(out_dir, 'ctx_path.pth')
    token_files_ctx_path = [os.path.join(out_dir, "train.before_ctx_path"), os.path.join(out_dir, "val.before_ctx_path")]
    build_label_vocab(token_files_ctx_path, ctx_path_vocab_file, min_appearance=args.min_appearance, max_size=args.max_vocab_size, is_path=True)
    ctx_path_vocab = Vocab(filename=ctx_path_vocab_file,
                        data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    torch.save(ctx_vocab, ctx_path_vocab_object)
    print('context path vocabulary size : {}'.format(ctx_path_vocab.size()))

    return src_vocab, tgt_vocab, ctx_vocab, ctx_path_vocab


def create_data_sets(split_name, src_vocab, tgt_vocab, ctx_vocab, ctx_path_vocab, dest, args):
    ctx_bos = ctx_vocab.getIndex(Constants.BOS_WORD)
    ctx_eos = ctx_vocab.getIndex(Constants.EOS_WORD)

    ctx_path_bos = ctx_path_vocab.getIndex(Constants.BOS_WORD)
    ctx_path_eos = ctx_path_vocab.getIndex(Constants.EOS_WORD)

    with open(os.path.join(args.out_dir, split_name) + ".src", "r") as fl:
        src_lines = fl.readlines()
    with open(os.path.join(args.out_dir, split_name) + ".dst", "r") as fl:
        tgt_lines = fl.readlines() + ['']
    with open(os.path.join(args.out_dir, split_name) + ".before_ctx", "r") as fl:
        before_ctx_lines = fl.readlines() + ['']
    with open(os.path.join(args.out_dir, split_name) + ".after_ctx", "r") as fl:
        after_ctx_lines = fl.readlines() + ['']
    with open(os.path.join(args.out_dir, split_name) + ".before_changes", "r") as fl:
        before_changes_lines = fl.readlines() + ['']
    with open(os.path.join(args.out_dir, split_name) + ".after_changes", "r") as fl:
        after_changes_lines = fl.readlines() + ['']
    with open(os.path.join(args.out_dir, split_name) + ".before_ctx_before", "r") as fl:
        before_ctx_before_lines = fl.readlines() + ['']
    with open(os.path.join(args.out_dir, split_name) + ".before_ctx_after", "r") as fl:
        before_ctx_after_lines = fl.readlines() + ['']
    with open(os.path.join(args.out_dir, split_name) + ".after_ctx_before", "r") as fl:
        after_ctx_before_lines = fl.readlines() + ['']
    with open(os.path.join(args.out_dir, split_name) + ".after_ctx_after", "r") as fl:
        after_ctx_after_lines = fl.readlines() + ['']
    with open(os.path.join(args.out_dir, split_name) + ".before_ctx_path", "r") as fl:
        before_ctx_path_lines = fl.readlines() + ['']
    with open(os.path.join(args.out_dir, split_name) + ".after_ctx_path", "r") as fl:
        after_ctx_path_lines = fl.readlines() + ['']
    samples = list()
    for i in range(len(src_lines)):
        src_tokens = src_lines[i]
        tgt_tokens = tgt_lines[i]
        before_ctx_tokens = before_ctx_lines[i]
        after_ctx_tokens = after_ctx_lines[i]
        before_ctx_before_tokens = before_ctx_before_lines[i]
        after_ctx_before_tokens = after_ctx_before_lines[i]
        before_ctx_after_tokens = before_ctx_after_lines[i]
        after_ctx_after_tokens = after_ctx_after_lines[i]
        before_changes_tokens = before_changes_lines[i]
        after_changes_tokens = after_changes_lines[i]
        before_ctx_path_tokens = process_paths(before_ctx_path_lines[i].rstrip('\n'))
        after_ctx_path_tokens = process_paths(after_ctx_path_lines[i].rstrip('\n'))
        src = [src_vocab.getIndex(t.strip(), default=Constants.UNK_WORD) for t in src_tokens.split()]
        tgt = [tgt_vocab.getIndex(t.strip(), default=Constants.UNK_WORD) for t in tgt_tokens.split()]
        before_ctx = [ctx_bos] + [ctx_vocab.getIndex(t.strip(), default=Constants.UNK_WORD) for t in before_ctx_tokens.split()] + [ctx_eos]
        after_ctx = [ctx_bos] + [ctx_vocab.getIndex(t.strip(), default=Constants.UNK_WORD) for t in after_ctx_tokens.split()] + [ctx_eos]

        before_ctx_before = [ctx_bos] + [ctx_vocab.getIndex(t.strip(), default=Constants.UNK_WORD) for t in before_ctx_before_tokens.split()] + [ctx_eos]
        after_ctx_before = [ctx_bos] + [ctx_vocab.getIndex(t.strip(), default=Constants.UNK_WORD) for t in after_ctx_before_tokens.split()] + [ctx_eos]

        before_ctx_after = [ctx_bos] + [ctx_vocab.getIndex(t.strip(), default=Constants.UNK_WORD) for t in before_ctx_after_tokens.split()] + [ctx_eos]
        after_ctx_after = [ctx_bos] + [ctx_vocab.getIndex(t.strip(), default=Constants.UNK_WORD) for t in after_ctx_after_tokens.split()] + [ctx_eos]

        before_changes = [ctx_bos] + [ctx_vocab.getIndex(t.strip(), default=Constants.UNK_WORD) for t in before_changes_tokens.split()] + [ctx_eos]
        after_changes = [ctx_bos] + [ctx_vocab.getIndex(t.strip(), default=Constants.UNK_WORD) for t in after_changes_tokens.split()] + [ctx_eos]

        before_ctx_path = [ctx_path_bos] + [ctx_path_vocab.getIndex(t.strip(), default=Constants.UNK_WORD) for t in before_ctx_path_tokens.split()] + [ctx_path_eos]
        after_ctx_path = [ctx_path_bos] + [ctx_path_vocab.getIndex(t.strip(), default=Constants.UNK_WORD) for t in after_ctx_path_tokens.split()] + [ctx_path_eos]
        samples.append({
            'src': src,
            'tgt': tgt,
            'before_ctx': before_ctx,
            'after_ctx': after_ctx,
            'before_ctx_before': before_ctx_before,
            'after_ctx_before': after_ctx_before,
            'before_ctx_after': before_ctx_after,
            'after_ctx_after': after_ctx_after,
            'before_ctx_changes': before_changes,
            'after_ctx_changes': after_changes,
            'before_ctx_path': before_ctx_path,
            'after_ctx_path': after_ctx_path,
            'original_tgt': tgt_tokens.strip()
        })
    torch.save(samples, dest)


if __name__ == '__main__':
    args = parse_args()
    print("Creating vocabulary")
    src_vocab, tgt_vocab, ctx_vocab, ctx_path_vocab = create_vocabularies(args.out_dir, args)
    print("Creating data pickles")
    print("Train")
    create_data_sets('train', src_vocab, tgt_vocab, ctx_vocab, ctx_path_vocab, os.path.join(args.out_dir, 'train.pth'), args)
    print("Dev")
    create_data_sets('val', src_vocab, tgt_vocab, ctx_vocab, ctx_path_vocab, os.path.join(args.out_dir, 'dev.pth'), args)
    print("Test")
    create_data_sets('test',  src_vocab, tgt_vocab, ctx_vocab, ctx_path_vocab, os.path.join(args.out_dir, 'test.pth'), args)
    print("Done")
