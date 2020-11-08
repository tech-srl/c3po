import os
import multiprocessing as mp
from tqdm import tqdm
import sys
import json
from typing import Sequence, Text
import scipy.sparse
import collections
import numpy as np
import tagging_converter
import tagging
import re

KEEP = "KEEP"
DELETE = "DELETE"
SWAP = "SWAP"

def _compute_lcs(source, target):
  """Computes the Longest Common Subsequence (LCS).

  Description of the dynamic programming algorithm:
  https://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    source: List of source tokens.
    target: List of target tokens.

  Returns:
    List of tokens in the LCS.
  """
  table = _lcs_table(source, target)
  return _backtrack(table, source, target, len(source), len(target))


def _lcs_table(source, target):
  """Returns the Longest Common Subsequence dynamic programming table."""
  rows = len(source)
  cols = len(target)
  lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
  for i in range(1, rows + 1):
    for j in range(1, cols + 1):
      if source[i - 1] == target[j - 1]:
        lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
      else:
        lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
  return lcs_table


def _backtrack(table, source, target, i, j):
  """Backtracks the Longest Common Subsequence table to reconstruct the LCS.

  Args:
    table: Precomputed LCS table.
    source: List of source tokens.
    target: List of target tokens.
    i: Current row index.
    j: Current column index.

  Returns:
    List of tokens corresponding to LCS.
  """
  if i == 0 or j == 0:
    return []
  if source[i - 1] == target[j - 1]:
    # Append the aligned token to output.
    return _backtrack(table, source, target, i - 1, j - 1) + [target[j - 1]]
  if table[i][j - 1] > table[i - 1][j]:
    return _backtrack(table, source, target, i, j - 1)
  else:
    return _backtrack(table, source, target, i - 1, j)


def _get_added_phrases(source: Text, target: Text) -> Sequence[Text]:
  """Computes the phrases that need to be added to the source to get the target.

  This is done by aligning each token in the LCS to the first match in the
  target and checking which phrases in the target remain unaligned.

  TODO(b/142853960): The LCS tokens should ideally be aligned to consecutive
  target tokens whenever possible, instead of aligning them always to the first
  match. This should result in a more meaningful phrase vocabulary with a higher
  coverage.

  Note that the algorithm is case-insensitive and the resulting phrases are
  always lowercase.

  Args:
    source: Source text.
    target: Target text.

  Returns:
    List of added phrases.
  """
  source_tokens = source.lower().split()
  target_tokens = target.lower().split()
  kept_tokens = _compute_lcs(source_tokens, target_tokens)
  added_phrases = []
  # Index of the `kept_tokens` element that we are currently looking for.
  kept_idx = 0
  phrase = []
  for token in target_tokens:
    if kept_idx < len(kept_tokens) and token == kept_tokens[kept_idx]:
      kept_idx += 1
      if phrase:
        added_phrases.append(' '.join(phrase))
        phrase = []
    else:
      phrase.append(token)
  if phrase:
    added_phrases.append(' '.join(phrase))
  return added_phrases


def _added_token_counts(data_iterator, try_swapping=False):
  """Computes how many times different phrases have to be added.

  Args:
    data_iterator: Iterator to yield source lists and targets. See function
      yield_sources_and_targets in utils.py for the available iterators. The
      strings in the source list will be concatenated, possibly after swapping
      their order if swapping is enabled.
    try_swapping: Whether to try if swapping sources results in less added text.
  Returns:
    Tuple (collections.Counter for phrases, added phrases for each example).
  """
  phrase_counter = collections.Counter()
  num_examples = 0
  all_added_phrases = []
  for sources, target in data_iterator:
    # logging.log_every_n(logging.INFO, f'{num_examples} examples processed.',
    #                     1000)
    added_phrases = _get_added_phrases(' '.join(sources), target)
    if try_swapping and len(sources) == 2:
      added_phrases_swap = _get_added_phrases(' '.join(sources[::-1]), target)
      # If we can align more and have to add less after swapping, we assume that
      # the sources would be swapped during conversion.
      if len(''.join(added_phrases_swap)) < len(''.join(added_phrases)):
        added_phrases = added_phrases_swap

    for phrase in added_phrases:
      phrase_counter[phrase] += 1
    all_added_phrases.append(added_phrases)
    num_examples += 1
  # logging.info(f'{num_examples} examples processed.\n')
  return phrase_counter, all_added_phrases


def yield_source_target(projects, samples_path):
    src_tgt_pairs = [((os.path.join(samples_path, p, p + ".before_normalized_filtered")), (os.path.join(samples_path, p, p + ".after_normalized_filtered"))) for p in os.listdir(samples_path) if p in projects]
    for src, tgt in src_tgt_pairs:
        with open(src, "r") as s:
            with open(tgt, "r") as t:
                t_lines = t.readlines()
                s_lines = s.readlines()
                for i in range(len(s_lines)):
                    yield [s_lines[i].strip()], t_lines[i].strip()


def create_vocab(splits, samples_path, out_path, vocabulary_size=500, num_extra_statistics=100, overwrite=True ,enable_swap=False):
    vocab_out_path = os.path.join(out_path, "vocab.txt")
    if os.path.exists(vocab_out_path) and overwrite is False:
        with open(vocab_out_path, "r") as f:
            return set(map(lambda s: s.strip(), f.readlines()))
    data_iterator = yield_source_target(splits['dev'] + splits['train'], samples_path)
    phrase_counter, all_added_phrases = _added_token_counts(
        data_iterator, try_swapping=enable_swap)
    phrases_list = [KEEP, DELETE]
    if enable_swap:
        phrases_list.append(SWAP)
    for i, (phrase, count) in enumerate(
            phrase_counter.most_common(vocabulary_size + num_extra_statistics)):
        if i < vocabulary_size:
            phrases_list.append(KEEP + "|" + phrase)
            phrases_list.append(DELETE + "|" + phrase)

    with open(vocab_out_path, "w") as f:
        f.writelines(map(lambda s: s + "\n", phrases_list))
    return set(phrases_list)


def create_samples_task(converter, iterator):
    samples = list()
    failure_samples = list()
    filtered_idx = list()
    for i, (src, tgt) in enumerate(iterator):
        task = tagging.EditingTask(src)
        tags = converter.compute_tags(task, tgt)
        tags = " ".join((map(lambda t:  str(t).replace(" ", "_"), tags)))
        src = src[0]
        if tags == '':
            failure_samples.append((src, tgt))
        else:
            samples.append((src, tags, tgt))
            filtered_idx.append(i)
    return samples, failure_samples, filtered_idx


def get_phrase_vocabulary_from_label_map(phrases_set):
    phrase_vocabulary = set()
    for label in phrases_set:
        tag = tagging.Tag(label)
        if tag.added_phrase:
            phrase_vocabulary.add(tag.added_phrase)
    return phrase_vocabulary


def extract_filtered(samples_path, projects, filtered_idx):
    res = {
    'before_ctx': list(),
    'after_ctx': list(),
    'before_ctx_before': list(),
    'before_ctx_after': list(),
    'after_ctx_before': list(),
    'after_ctx_after': list(),
    'before_changes': list(),
    'after_changes': list(),
    'before_ctx_path': list(),
    'after_ctx_path': list()
    }
    changes_pattern = re.compile(r"<%>.*?</%>")
    for p in os.listdir(samples_path):
        if p in projects:
            with open(os.path.join(samples_path, p, p + ".before_ctx_filtered"), "r") as f:
                before_ctx = list(map(lambda x: x.strip(), f.readlines()))
                before_ctx_changes = list()
                for bf in before_ctx:
                    before_ctx_changes.append(" ".join(map(lambda s: s.strip(), changes_pattern.findall(bf))))
                res['before_ctx'] += before_ctx
                res['before_changes'] += before_ctx_changes
            with open(os.path.join(samples_path, p, p + ".after_ctx_filtered"), "r") as f:
                after_ctx = list(map(lambda x: x.strip(), f.readlines()))
                after_ctx_changes = list()
                for af in after_ctx:
                    after_ctx_changes.append(" ".join(map(lambda s: s.strip(), changes_pattern.findall(af))))
                res['after_ctx'] += after_ctx
                res['after_changes'] += after_ctx_changes
            with open(os.path.join(samples_path, p, p + ".before_ctx_before_normalized_filtered"), "r") as f:
                res['before_ctx_before'] += list(map(lambda x: x.strip(), f.readlines()))
            with open(os.path.join(samples_path, p, p + ".after_ctx_before_normalized_filtered"), "r") as f:
                res['after_ctx_before'] += list(map(lambda x: x.strip(), f.readlines()))
            with open(os.path.join(samples_path, p, p + ".before_ctx_after_normalized_filtered"), "r") as f:
                res['before_ctx_after'] += list(map(lambda x: x.strip(), f.readlines()))
            with open(os.path.join(samples_path, p, p + ".after_ctx_after_normalized_filtered"), "r") as f:
                res['after_ctx_after'] += list(map(lambda x: x.strip(), f.readlines()))
            with open(os.path.join(samples_path, p, p + ".before_ctx_path"), "r") as f:
                res['before_ctx_path'] += list(map(lambda x: x.strip(), f.readlines()))
            with open(os.path.join(samples_path, p, p + ".after_ctx_path"), "r") as f:
                res['after_ctx_path'] += list(map(lambda x: x.strip(), f.readlines()))
    for k in res.keys():
        res[k] = [res[k][i] for i in filtered_idx]
    return res


def dump_ctx_dict(ctx_dict, split_name, out_path):
    for k in ctx_dict.keys():
        with open(os.path.join(out_path, split_name + "." + k), "w") as f:
            f.write("\n".join(ctx_dict[k]))


def create_samples(splits, samples_path, out_path, phrases_set):
    converter = tagging_converter.TaggingConverter(get_phrase_vocabulary_from_label_map(phrases_set), do_swap=True)

    train_iterator = yield_source_target(splits['train'], samples_path)
    train_samples, train_failure_samples, train_filtered_idx = create_samples_task(converter, train_iterator)
    train_ctx_dict = extract_filtered(samples_path, splits['train'], train_filtered_idx)

    val_iterator = yield_source_target(splits['dev'], samples_path)
    val_samples, val_failure_samples, val_filtered_idx = create_samples_task(converter, val_iterator)
    val_ctx_dict = extract_filtered(samples_path, splits['dev'], val_filtered_idx)

    test_iterator = yield_source_target(splits['test'], samples_path)
    test_samples, test_failure_samples, test_filtered_idx = create_samples_task(converter, test_iterator)
    test_ctx_dict = extract_filtered(samples_path, splits['test'], test_filtered_idx)

    print("Train: {}, Val: {}, Test: {}".format(len(train_samples), len(val_samples), len(test_samples)))
    # print("Failed samples: Train: {}, Val: {}, Test: {}, Total: {}".format(len(train_failure_samples), len(val_failure_samples), len(test_failure_samples), len(train_failure_samples) + len(val_failure_samples) + len(test_failure_samples)))
    train_src, train_dst, train_original_dst = zip(*train_samples)
    val_src, val_dst, val_original_dst = zip(*val_samples)
    test_src, test_dst, test_original_dst = zip(*test_samples)

    train_failure_samples_src, train_failure_samples_dst = zip(*train_failure_samples)
    val_failure_samples_src, val_failure_samples_dst = zip(*val_failure_samples)
    test_failure_samples_src, test_failure_samples_dst = zip(*test_failure_samples)
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

    with open(os.path.join(out_path, "train.original_dst"), "w") as f:
        f.write("\n".join(train_original_dst))
    with open(os.path.join(out_path, "val.original_dst"), "w") as f:
        f.write("\n".join(val_original_dst))
    with open(os.path.join(out_path, "test.original_dst"), "w") as f:
        f.write("\n".join(test_original_dst))

    with open(os.path.join(out_path, "train.failed_src"), "w") as f:
        f.write("\n".join(train_failure_samples_src))
    with open(os.path.join(out_path, "val.failed_src"), "w") as f:
        f.write("\n".join(val_failure_samples_src))
    with open(os.path.join(out_path, "test.failed_src"), "w") as f:
        f.write("\n".join(test_failure_samples_src))

    with open(os.path.join(out_path, "train.failed_dst"), "w") as f:
        f.write("\n".join(train_failure_samples_dst))
    with open(os.path.join(out_path, "val.failed_dst"), "w") as f:
        f.write("\n".join(val_failure_samples_dst))
    with open(os.path.join(out_path, "test.failed_dst"), "w") as f:
        f.write("\n".join(test_failure_samples_dst))

    with open(os.path.join(out_path, "train.projects"), "w") as f:
        f.write("\n".join(splits['train']))
    with open(os.path.join(out_path, "val.projects"), "w") as f:
        f.write("\n".join(splits['dev']))
    with open(os.path.join(out_path, "test.projects"), "w") as f:
        f.write("\n".join(splits['test']))

    dump_ctx_dict(train_ctx_dict, 'train', out_path)
    dump_ctx_dict(val_ctx_dict, 'val', out_path)
    dump_ctx_dict(test_ctx_dict, 'test', out_path)


if __name__ == '__main__':
    samples_path = os.path.abspath(sys.argv[1])
    split_json = os.path.abspath(sys.argv[2])
    out_path = os.path.abspath(sys.argv[3])
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    vocabulary_size = 520
    num_extra_statistics = 100
    enable_swap = True

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    with open(split_json, "r") as f:
        splits = json.load(f)

    print("Creating phases vocabulary...", file=sys.stderr)
    phrases_set = create_vocab(splits, samples_path, out_path, vocabulary_size=vocabulary_size, overwrite=True, enable_swap=enable_swap)

    print("Creating datasets...", file=sys.stderr)
    create_samples(splits, samples_path, out_path, phrases_set)
    print("Done!", file=sys.stderr)