from __future__ import division
from __future__ import print_function

import os
import math
import re
import torch
from vocab import Vocab
import Constants
from tqdm import tqdm

# loading GLOVE word vectors
# if .pth file is found, will load that
# else will load from .txt file & save
def load_word_vectors(path):
    if os.path.isfile(path + '.pth') and os.path.isfile(path + '.vocab'):
        print('==> File found, loading to memory')
        vectors = torch.load(path + '.pth')
        vocab = Vocab(filename=path + '.vocab')
        return vocab, vectors
    # saved file not found, read from txt file
    # and create tensors for word vectors
    print('==> File not found, preparing, be patient')
    count = sum(1 for line in open(path + '.txt', 'r', encoding='utf8', errors='ignore'))
    with open(path + '.txt', 'r') as f:
        contents = f.readline().rstrip('\n').split(' ')
        dim = len(contents[1:])
    words = [None] * (count)
    vectors = torch.zeros(count, dim, dtype=torch.float, device='cpu')
    with open(path + '.txt', 'r', encoding='utf8', errors='ignore') as f:
        idx = 0
        for line in f:
            contents = line.rstrip('\n').split(' ')
            words[idx] = contents[0]
            values = list(map(float, contents[1:]))
            vectors[idx] = torch.tensor(values, dtype=torch.float, device='cpu')
            idx += 1
    with open(path + '.vocab', 'w', encoding='utf8', errors='ignore') as f:
        for word in words:
            f.write(word + '\n')
    vocab = Vocab(filename=path + '.vocab')
    torch.save(vectors, path + '.pth')
    return vocab, vectors


# write unique words from a set of files to a new file
def build_path_vocabs(tokens_filenames, path_vocab_file=None, src_tgt_vocab_file=None, position_vocab_file=None, ignore_directions=True, min_appearance=0, max_size=None):
    def aux(count_dict, vocab, min_appearance, tokens):
        for token in tokens:
            if token == '':
                continue
            count_dict[token] = 1 if token not in count_dict.keys() else count_dict[token] + 1
            if count_dict[token] >= min_appearance:
                vocab.add(token)

    def limit_size(vocab_count, size):
        sorted_vocab = [(count, label) for label, count in vocab_count.items()]
        sorted(sorted_vocab, key=lambda t: t[0], reverse=True)
        sorted_vocab = sorted_vocab[:size]
        _, sorted_vocab = zip(*sorted_vocab)
        sorted_vocab = set(sorted_vocab)
        return sorted_vocab

    path_vocab = set()
    src_tgt_vocab = set()
    position_vocab = set()
    path_count = dict()
    src_tgt_count = dict()
    position_count = dict()
    if tokens_filenames is not None:
        for filename in tqdm(tokens_filenames):
            with open(filename, 'r') as f:
                for line in f:
                    for path in line.split("\t"):
                        tokens = [t.strip() for t in path.split(" ") if t.strip() != Constants.DOWN_WORD and t.strip() != Constants.UP_WORD]
                        positions = tokens[1::2]
                        tokens = tokens[0::2]
                        src = tokens[0].split("|")
                        aux(src_tgt_count, src_tgt_vocab, min_appearance, src)
                        tgt = tokens[-1].split("|")
                        aux(src_tgt_count, src_tgt_vocab, min_appearance, tgt)
                        tokens = tokens[1:-1]
                        aux(path_count, path_vocab, 0, tokens)
                        aux(position_count, position_vocab, min_appearance, positions)
    if not ignore_directions:
        path_vocab.add(Constants.DOWN_WORD)
        path_vocab.add(Constants.UP_WORD)
    if path_vocab_file is not None:
        with open(path_vocab_file, 'w') as f:
            for token in sorted(path_vocab):
                f.write(token + '\n')
    if max_size is not None:
        src_tgt_vocab = limit_size(src_tgt_count, max_size)
    # limit position vocab to be at most 50
    position_vocab = limit_size(position_count, 50)

    if src_tgt_vocab_file is not None:
        with open(src_tgt_vocab_file, 'w') as f:
            for token in sorted(src_tgt_vocab):
                f.write(token + '\n')
    if position_vocab_file is not None:
        with open(position_vocab_file, 'w') as f:
            for token in sorted(position_vocab):
                f.write(token + '\n')
    return path_vocab, src_tgt_vocab, position_vocab


def build_label_vocab(tokens_filenames, vocabfile=None, min_appearance=0, max_size=None):
    vocab = set()
    label_count = dict()
    if tokens_filenames is not None:
        for filename in tqdm(tokens_filenames):
            with open(filename, 'r') as f:
                for line in f:
                    tokens = [t.strip() for t in line.rstrip('\n').split(" ")][0::2]
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

# mapping from scalar to vector
def map_label_to_target(label, num_classes):
    target = torch.zeros(1, num_classes, dtype=torch.float, device='cpu')
    ceil = int(math.ceil(label))
    floor = int(math.floor(label))
    if ceil == floor:
        target[0, floor-1] = 1
    else:
        target[0, floor-1] = ceil - label
        target[0, ceil-1] = label - floor
    return target

def get_num_of_lines(filename):
    with open(filename, 'r', encoding='utf8', errors='ignore') as f:
        count = len(f.readlines())
        return count


def get_sub_tokens(token):
    return set(token.split("|"))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)