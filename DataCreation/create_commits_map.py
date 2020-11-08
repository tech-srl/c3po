from difflib import unified_diff
import pickle
import os
from diff_symbolizer import *
from csharp_tokenizer import *
from tqdm import tqdm
import multiprocessing as mp
import sys
import json


def get_diffs(before, after, context_size=0, diff_size_limit=None):
    pattern = r"@@ -([0-9]+)(,[0-9]+)? \+([0-9]+)(,[0-9]+)? @@"
    regex = re.compile(pattern)
    before_list = list(
        map(lambda x: x.replace('\t', ''), filter(lambda x: not re.match(r'^\s*$', x), before.splitlines())))
    after_list = list(
        map(lambda x: x.replace('\t', ''), filter(lambda x: not re.match(r'^\s*$', x), after.splitlines())))
    multiline_diff = 0
    inline_diff = 0
    diff = list(unified_diff(before_list, after_list, n=context_size))
    diff.reverse()
    counter_plus = 0
    counter_minus = 0
    res = list()
    for l in diff:
        if l.startswith("---") or l.startswith("+++"):
            continue
        if l.startswith("@"):
            counter = abs(counter_plus - counter_minus)
            if diff_size_limit is not None and counter > diff_size_limit:
                counter_plus = 0
                counter_minus = 0
                continue
            before_start = 0
            before_length = 1
            after_start = 0
            after_length = 1
            m = regex.search(l)
            if m.group(1) is not None:
                before_start = int(m.group(1)) - 1
            if m.group(2) is not None:
                before_length = int(m.group(2)[1:])
            if m.group(3) is not None:
                after_start = int(m.group(3)) - 1
            if m.group(4) is not None:
                after_length = int(m.group(4)[1:])
            if after_length == 0:
                after_start += 1
            if before_length == 0:
                before_start += 1
            before_str = "\n".join(before_list[before_start: before_start + before_length])
            after_str = "\n".join(after_list[after_start: after_start + after_length])
            res.append({
                'before_line_num': before_start,
                'after_line_num': after_start,
                'before_length': before_length,
                'after_length': after_length,
                'before': before_str,
                'after': after_str,
            })
            if before_length == 1 and after_length == 1:
                inline_diff += 1
            else:
                multiline_diff += 1
            counter_plus = 0
            counter_minus = 0
        if l.startswith("+"):
            counter_plus += 1
        elif l.startswith("-"):
            counter_minus += 1
    return res, inline_diff, multiline_diff, "\n".join(before_list), "\n".join(after_list)


def create_combined_file(before, after, diffs):
    before = before.splitlines()
    after = after.splitlines()
    diffs = reversed(diffs)
    res = list()
    before_line = 0
    after_line = 0
    for diff in diffs:
        if diff['before_line_num'] > before_line or diff['after_line_num'] > after_line:
            res.append({
                'before_line_num': before_line,
                'after_line_num': after_line,
                'before_length': diff['before_line_num'] - before_line,
                'after_length': diff['after_line_num'] - after_line,
                'before': before[before_line:diff['before_line_num']],
                'after': after[after_line:diff['after_line_num']],
            })
        res.append({
            'before_line_num': diff['before_line_num'],
            'after_line_num': diff['after_line_num'],
            'before_length': diff['before_length'],
            'after_length': diff['after_length'],
            'before': diff['before'].splitlines(),
            'after': diff['after'].splitlines(),
            })
        before_line = diff['before_line_num'] + diff['before_length']
        after_line = diff['after_line_num'] + diff['after_length']
    if before_line < len(before) or after_line < len(after):
        res.append({
            'before_line_num': before_line,
            'after_line_num': after_line,
            'before_length': len(before) - before_line,
            'after_length': len(after) - after_line,
            'before': before[before_line:],
            'after': after[after_line:]
        })
    return res


def _group_entries(group):
    group = list(group)
    for g in group:
        if g['identical'] is False:
            return group
    res = None
    for i, entry in enumerate(group):
        if i == 0:
            res = entry
            continue
        res['normalized_before'] += entry['normalized_before']
        res['normalized_after'] += entry['normalized_after']
        res['before_length'] += entry['before_length']
        res['after_length'] += entry['after_length']
    return [res]


def agument_entry(entry, max_length=128):
    try:
        entry['error'] = False
        entry['normalized_before'] = list()
        entry['normalized_after'] = list()
        entry['diff_symbols'] = None
        entry['integrated_diff_symbol'] = None
        entry['identical'] = True
        for before_str in entry['before']:
            normalized_before, before_seprated, before_tokens = tokenize_code(before_str)
            if normalized_before is None:
                raise ValueError
            entry['normalized_before'].append(normalized_before)
        for after_str in entry['after']:
            normalized_after, after_seprated, after_tokens = tokenize_code(after_str)
            if normalized_after is None:
                raise ValueError
            entry['normalized_after'].append(normalized_after)
        entry['identical'] = entry['normalized_before'] == entry['normalized_after']
        if entry['identical'] is False:
            entry['diff_symbols'] = list()
            entry['integrated_diff_symbol'] = list()

            for l in entry['normalized_before']:
                if len(l.split()) > max_length:
                    raise ValueError
            for l in entry['normalized_after']:
                if len(l.split()) > max_length:
                    raise ValueError
            try:
                # align lines
                if len(entry['normalized_before']) != len(entry['normalized_after']):
                    if len(entry['normalized_before']) > max_length or len(entry['normalized_after']) > max_length:
                        raise ValueError
                    entry['normalized_before'], entry['normalized_after'] = align_lines(entry['normalized_before'],
                                                                                        entry['normalized_after'])
                # add diff symbols
                for i in range(len(entry['normalized_before'])):
                    diff_symbols, integrated_diff_symbol_str = create_diff_strs(entry['normalized_before'][i],
                                                                                entry['normalized_after'][i])
                    entry['diff_symbols'].append(diff_symbols)
                    entry['integrated_diff_symbol'].append(integrated_diff_symbol_str)
            except MemoryError:
                # can happened on large inputs
                raise ValueError
            # check identifty after padding
            entry['identical'] = entry['normalized_before'] == entry['normalized_after']
    except ValueError:
        entry['error'] = True


def augment_combined(combined, max_length=128):
    for entry in combined:
        agument_entry(entry, max_length)
    # merge entries with no changes:
    result = list()
    for k, g in itertools.groupby(combined, lambda e: e['identical']):
        result += _group_entries(g)
    return result


def create_commit_map_task(args):
    path, out_path, idx = args
    try:
        with open(path, "r") as f:
            entry = json.load(f)
    except:
        return None, None
    id_parts = entry['id'].split('|')
    commit_id = "|".join(id_parts[:-1])
    file_path = id_parts[-1]
    diffs, inline_diff, multiline_diff, before, after = get_diffs(entry['prev_file'], entry['updated_file'],
                                                                  context_size=0)
    combined = create_combined_file(before, after, diffs)
    combined = augment_combined(combined)
    file_entry = {
        'path': file_path,
        # 'before': before,
        # 'after': after,
        # 'diffs': diffs,
        'inline_diffs': inline_diff,
        'multiline_diffs': multiline_diff,
        'combined': combined
    }
    entry_path = os.path.join(out_path, commit_id + "_" + str(idx) + ".pickle")
    with open(entry_path, "wb") as f:
        pickle.dump(file_entry, f, protocol=pickle.HIGHEST_PROTOCOL)

    return commit_id, entry_path


def aggregation_task(args):
    k, v, out_path = args
    files_entry = list()
    for entry_path in v['files']:
        with open(entry_path, "rb") as f:
            files_entry.append(pickle.load(f))
        os.remove(entry_path)
    v['files'] = files_entry
    path = os.path.join(out_path, "{}.pickle".format(k))
    with open(path, "wb") as f:
        pickle.dump(v, f, protocol=pickle.HIGHEST_PROTOCOL)
    return k


def create_commits_map(data_path, out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    commits_map = dict()
    print("Listing files...", file=sys.stderr)
    files = list(map(lambda t: (os.path.join(data_path, t[1]), out_path, t[0]), enumerate(os.listdir(data_path))))
    print("Processing...", file=sys.stderr)
    i = 1
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        with tqdm(total=len(files)) as pbar:
            for commit_id, entry_path in pool.imap(create_commit_map_task, files):
                pbar.update()
                if commit_id is None:
                    i += 1
                    continue
                if commit_id not in commits_map:
                    commits_map[commit_id] = dict()
                    commits_map[commit_id]['files'] = list()
                    commits_map[commit_id]['num_of_files'] = 0
                commits_map[commit_id]['files'].append(entry_path)
                commits_map[commit_id]['num_of_files'] += 1
                if i % 100000 == 0:
                    print("{} iteration".format(i), file=sys.stderr)
                i += 1
        print("Pickling data...", file=sys.stderr)
        items = list(map(lambda item: (item[0], item[1], out_path), list(commits_map.items())))
        with tqdm(total=len(items)) as pbar:
            for _ in pool.imap(aggregation_task, items):
                pbar.update()
    #
    #
    # for k, v in tqdm(commits_map.items()):
    #     path = os.path.join(out_path, "{}.pickle".format(k))
    #     with open(path, "wb") as f:
    #         pickle.dump(v, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done!", file=sys.stderr)


if __name__ == '__main__':
    commits_dir = os.path.abspath(sys.argv[1])
    proccesed_dir = os.path.abspath(sys.argv[2])

    if not os.path.exists(proccesed_dir):
        os.mkdir(proccesed_dir)
        
    create_commits_map(commits_dir, proccesed_dir)
