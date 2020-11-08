import pickle
import os
import multiprocessing as mp
import sys
from tqdm import tqdm
import re
from functools import reduce
import string
from diff_symbolizer import *
from csharp_tokenizer import tokenize_code_for_dataset


def get_context(combined, focus_idx, context_size):
    # before
    before_context_before = list()
    before_context_after = list()
    before_context_before_normalized = list()
    before_context_after_normalized = list()
    before_context = list()
    size_remaining = context_size
    found_change = False
    for i in reversed(range(0, focus_idx)):
        chunk = combined[i]
        if size_remaining == 0 or chunk['error'] is True:
            # before_context_before = chunk['before'][-size_remaining:] + before_context_before
            # before_context_after = chunk['after'][-size_remaining:] + before_context_after
            break
        # handle move tokens from end of the line, to the beginning of the next line
        if chunk['identical'] is False and " ".join(chunk['normalized_before']) != " ".join(chunk['normalized_after']):
            found_change = True
        chunk_code = chunk['normalized_before'] if chunk['identical'] is True else chunk['integrated_diff_symbol']
        chunk_code = list(filter(lambda x: x != '', chunk_code))
        chunk_code = chunk_code[-size_remaining:]
        before_context = chunk_code + before_context
        before_context_before = chunk['before'][-size_remaining:] + before_context_before
        before_context_after = chunk['after'][-size_remaining:] + before_context_after

        normalized_before = chunk['normalized_before']
        normalized_before = list(filter(lambda x: x != '', normalized_before))
        normalized_before = normalized_before[-size_remaining:]
        before_context_before_normalized = normalized_before + before_context_before_normalized

        normalized_after = chunk['normalized_after']
        normalized_after = list(filter(lambda x: x != '', normalized_after))
        normalized_after = normalized_after[-size_remaining:]
        before_context_after_normalized = normalized_after + before_context_after_normalized

        chunk_size = len(chunk_code)
        size_remaining -= chunk_size

    #after
    after_context_before = list()
    after_context_after = list()
    after_context_before_normalized = list()
    after_context_after_normalized = list()
    after_context = list()
    size_remaining = context_size
    for i in range(focus_idx+1, len(combined)):
        chunk = combined[i]
        if size_remaining == 0 or chunk['error'] is True:
            # after_context_before += chunk['before'][:size_remaining]
            # after_context_after += chunk['after'][:size_remaining]
            break
        # handle move tokens from end of the line, to the beginning of the next lint
        #found_change = found_change or not chunk['identical']
        if chunk['identical'] is False and " ".join(chunk['normalized_before']) != " ".join(chunk['normalized_after']):
            found_change = True
        chunk_code = chunk['normalized_before'] if chunk['identical'] is True else chunk['integrated_diff_symbol']
        chunk_code = list(filter(lambda x: x != '', chunk_code))
        chunk_code = chunk_code[:size_remaining]
        after_context += chunk_code
        after_context_before += chunk['before'][:size_remaining]
        after_context_after += chunk['after'][:size_remaining]

        normalized_before = chunk['normalized_before']
        normalized_before = list(filter(lambda x: x != '', normalized_before))
        normalized_before = normalized_before[:size_remaining]
        after_context_before_normalized += normalized_before

        normalized_after = chunk['normalized_after']
        normalized_after = list(filter(lambda x: x != '', normalized_after))
        normalized_after = normalized_after[:size_remaining]
        after_context_after_normalized += normalized_after

        chunk_size = len(chunk_code)
        size_remaining -= chunk_size
    res = {
        'before_context': before_context,
        'after_context': after_context,
        'before_context_before': before_context_before,
        'before_context_after': before_context_after,
        'after_context_before': after_context_before,
        'after_context_after': after_context_after,
        'before_context_before_normalized': before_context_before_normalized,
        'before_context_after_normalized': before_context_after_normalized,
        'after_context_before_normalized': after_context_before_normalized,
        'after_context_after_normalized': after_context_after_normalized,
        'found_change': found_change
    }
    return res


def merge_changes(lines):
    add_regex = re.compile(r"(<%> <\+> (.*?) </%>)")
    del_regex = re.compile(r"(<%> <-> (.*?) </%>)")
    regexs = [(add_regex, '<+>'), (del_regex, '<->')]
    sub_regex = re.compile(r"(<%> <\*> (.+?) -> (.+?) </%>)")
    i = 0
    while i < len(lines)-1:
        treated = False
        for reg, symbol in regexs:
            change_a = reg.findall(lines[i])
            if len(change_a) > 0 and lines[i].endswith(change_a[-1][0]):
                whole_a = change_a[-1][0]
                inside_a = change_a[-1][1]
                change_b = reg.findall(lines[i+1])
                if len(change_b) > 0 and lines[i+1].startswith(change_b[0][0]):
                    whole_b = change_b[0][0]
                    inside_b = change_b[0][1]
                    unified_change = "<%> " + symbol + " " + inside_a + " " + inside_b + " </%>"
                    lines[i] = lines[i][:-len(whole_a)] + unified_change
                    lines[i+1] = lines[i+1][len(whole_b):].strip()
                    treated = True
                    if lines[i + 1] == '':
                        del lines[i + 1]
                    i -= 1
                    break
                # <-> <+> or  <+> <-> becomes <*> ->
                reg = regexs[0][0] if regexs[0][1] != symbol else regexs[1][0]
                change_b = reg.findall(lines[i + 1])
                if len(change_b) > 0 and lines[i + 1].startswith(change_b[0][0]):
                    whole_b = change_b[0][0]
                    inside_b = change_b[0][1]
                    before, after = (inside_a, inside_b) if symbol == '<->' else (inside_b, inside_a)
                    unified_change = "<%> <*> " + before + " -> " + after + " </%>"
                    lines[i] = lines[i][:-len(whole_a)] + unified_change
                    lines[i + 1] = lines[i + 1][len(whole_b):].strip()
                    treated = True
                    if lines[i + 1] == '':
                        del lines[i + 1]
                    i -= 1
                    break
        if treated is False:
            change_a = sub_regex.findall(lines[i])
            if len(change_a) > 0 and lines[i].endswith(change_a[-1][0]):
                whole_a = change_a[-1][0]
                inside_aa = change_a[-1][1]
                inside_ab = change_a[-1][2]
                change_b = sub_regex.findall(lines[i + 1])
                if len(change_b) > 0 and lines[i + 1].startswith(change_b[0][0]):
                    whole_b = change_b[0][0]
                    inside_ba = change_b[0][1]
                    inside_bb = change_b[0][2]
                    unified_change = "<%> <*> " + inside_aa + " " + inside_ba + " -> " + inside_ab + " " + inside_bb + " </%>"
                    lines[i] = lines[i][:-len(whole_a)] + unified_change
                    lines[i + 1] = lines[i + 1][len(whole_b):].strip()
                    if lines[i + 1] == '':
                        del lines[i + 1]
                    i -= 1
        i += 1
    return lines


def extract_updates(before, after, tokens):
    changes = set()
    for i, token in enumerate(tokens):
        if token == "Token.Name":
            changes.add(before[i] + "<@@>" + after[i])
    return changes


def is_trivial(before_ctx, after_ctx, before, after):
    before_ctx_seperated, before_ctx_tokens = tokenize_code_for_dataset(before_ctx)
    after_ctx_seperated, after_ctx_tokens = tokenize_code_for_dataset(after_ctx)
    if before_ctx_tokens != after_ctx_tokens:
        return False
    before_seperated, before_tokens = tokenize_code_for_dataset(before)
    after_seperated, after_tokens = tokenize_code_for_dataset(after)
    if before_tokens != after_tokens:
        return False
    ctx_changes = extract_updates(before_ctx_seperated, after_ctx_seperated, before_ctx_tokens)
    focus_changes = extract_updates(before_seperated, after_seperated, before_tokens)
    if len(focus_changes) > 0 and focus_changes <= ctx_changes:
        return True
    return False


def is_unique(context, change):
    changes_pattern = re.compile(r"<%>(.*?)</%>")
    sample_changes_set = set(map(lambda s: s.strip(), changes_pattern.findall(" ".join(change))))
    context_changes_set = set(map(lambda s: s.strip(), changes_pattern.findall(" ".join(context))))
    return len(sample_changes_set & context_changes_set) == 0


def count_changes(change):
    changes_pattern = re.compile(r"<%>(.*?)</%>")
    changes_list = list(map(lambda s: s.strip(), changes_pattern.findall(" ".join(change))))
    return len(changes_list)


def get_changes(changes):
    exclude = set(string.punctuation)
    add_regex = re.compile(r"(<%> +<\+> (.*?) </%>)")
    del_regex = re.compile(r"(<%> <-> (.*?) </%>)")
    sub_regex = re.compile(r"(<%> <\*> (.+?) -> (.+?) </%>)")
    join_changes = " ".join(changes)
    added_changes_set = set(reduce(lambda x, y: x + y, map(lambda t: t[1].strip().split(), add_regex.findall(join_changes)), list()))
    del_changes_set = set(reduce(lambda x, y: x + y, map(lambda t: t[1].strip().split(), del_regex.findall(join_changes)), list()))
    sub_changes_list = list(map(lambda t: (t[1].strip().split(), t[2].strip().split()), sub_regex.findall(join_changes)))
    sub_del, sub_add = set(), set()
    if len(sub_changes_list) > 0:
        sub_del, sub_add = zip(*sub_changes_list)
        sub_del = reduce(lambda x, y: x + y, sub_del, list())
        sub_add = reduce(lambda x, y: x + y, sub_add, list())
    added_changes_set |= set(sub_add)
    del_changes_set |= set(sub_del)
    added_changes_set_excluded = set(filter(lambda x: x not in exclude, added_changes_set))
    del_changes_set_excluded = set(filter(lambda x: x not in exclude, del_changes_set))
    return added_changes_set_excluded, del_changes_set_excluded


def mov_sub_tree(changes, ctx_changes):
    added_changes_set, del_changes_set = get_changes(changes)
    # ctx_added_changes_set, ctx_del_changes_set = get_changes(ctx_changes)
    if len(added_changes_set & del_changes_set) == len(added_changes_set) > 0:
        return True
    return False


def create_sample_from_file(file_entry, inline_change_only, changed_context_only, merge_change_in_context, unique_change_only, non_trivial_change_only, min_changes, mov_in_change, overwrite_integarted_changes, context_size):
    res = list()
    for i, chunk in enumerate(file_entry['combined']):
        joined_normalized_before = " ".join(chunk['normalized_before'])
        joined_normalized_after = " ".join(chunk['normalized_after'])
        if chunk['error'] is True or chunk['identical'] is True or joined_normalized_before.strip() == '' or joined_normalized_before.strip() == joined_normalized_after.strip():
            continue
        if (inline_change_only is False and chunk['before_length'] > 0) or (inline_change_only is True and chunk['before_length'] == 1):# and chunk['after_length'] <= 1):
            context_dict = get_context(file_entry['combined'], i, context_size)
            before_context, after_context = context_dict['before_context'], context_dict['after_context']
            if (changed_context_only is False) or (changed_context_only is True and context_dict['found_change'] is True):
                change = chunk['integrated_diff_symbol']
                if overwrite_integarted_changes is True:
                    _, change = create_diff_strs(joined_normalized_before, joined_normalized_after)
                    change = [change]
                if merge_change_in_context is True:
                    before_context = merge_changes(before_context)
                    after_context = merge_changes(after_context)
                    change = merge_changes(change)
                before_ctx = "\n".join(context_dict['before_context_before'] + context_dict['after_context_before'])
                after_ctx = "\n".join(context_dict['before_context_after'] + context_dict['after_context_after'])
                if non_trivial_change_only is True and is_trivial(before_ctx, after_ctx, "\n".join(chunk['before']), "\n".join(chunk['after'])) is True:
                    # print("\n".join(before_context))
                    # print(50 * "*")
                    # print(chunk['before'])
                    # print(50 * "%")
                    # print(chunk['after'])
                    # print(50 * "*")
                    # print("\n".join(after_context))
                    # print(50 * "-")
                    continue
                if unique_change_only is True and is_unique(before_context + after_context, change) is False:
                    continue
                if min_changes is not None and count_changes(change) < min_changes:
                    continue
                if mov_in_change and not mov_sub_tree(change, before_context + after_context):
                    continue
                res.append({
                    'before_context': before_context,
                    'after_context': after_context,
                    'before_context_before': context_dict['before_context_before'],
                    'before_context_after': context_dict['before_context_after'],
                    'after_context_before': context_dict['after_context_before'],
                    'after_context_after': context_dict['after_context_after'],
                    'before_context_before_normalized': context_dict['before_context_before_normalized'],
                    'before_context_after_normalized': context_dict['before_context_after_normalized'],
                    'after_context_before_normalized': context_dict['after_context_before_normalized'],
                    'after_context_after_normalized': context_dict['after_context_after_normalized'],
                    'before': chunk['before'],
                    'after': chunk['after'],
                    'before_normalized': chunk['normalized_before'],
                    'after_normalized': chunk['normalized_after'],
                    'integrated_change': change
                })
    return res


def create_samples_task(args):
    dir_path, commit_path, inline_change_only, changed_context_only, merge_change_in_context, unique_change_only, non_trivial_change_only, min_changes, mov_in_change, overwrite_integarted_changes, context_size = args
    res = list()
    project_id = commit_path[:-1].split("|")[0]
    with open(os.path.join(dir_path, commit_path), "rb") as f:
        commit_map = pickle.load(f)
    flag = False
    for file_entry in commit_map['files']:
        curr_res = create_sample_from_file(file_entry, inline_change_only, changed_context_only, merge_change_in_context, unique_change_only, non_trivial_change_only, min_changes, mov_in_change, overwrite_integarted_changes, context_size)
        res += curr_res
    return res, project_id


def dump_samples(out_path, project_name, project_samples):
    out_dir = os.path.join(out_path, project_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    before_context = list(map(lambda sample: " ".join(sample['before_context']).strip(), project_samples))
    after_context = list(map(lambda sample: " ".join(sample['after_context']).strip(), project_samples))

    before_context_before = list(map(lambda sample: "\\n".join(sample['before_context_before']), project_samples))
    before_context_after = list(map(lambda sample: "\\n".join(sample['before_context_after']), project_samples))
    after_context_before = list(map(lambda sample: "\\n".join(sample['after_context_before']), project_samples))
    after_context_after = list(map(lambda sample: "\\n".join(sample['after_context_after']), project_samples))

    before_context_before_normalized = list(map(lambda sample: " ".join(sample['before_context_before_normalized']), project_samples))
    before_context_after_normalized = list(map(lambda sample: " ".join(sample['before_context_after_normalized']), project_samples))
    after_context_before_normalized = list(map(lambda sample: " ".join(sample['after_context_before_normalized']), project_samples))
    after_context_after_normalized = list(map(lambda sample: " ".join(sample['after_context_after_normalized']), project_samples))

    before = list(map(lambda sample: "\\n".join(sample['before']), project_samples))
    after = list(map(lambda sample: "\\n".join(sample['after']), project_samples))

    before_normalized = list(map(lambda sample: " ".join(sample['before_normalized']).strip(), project_samples))
    after_normalized = list(map(lambda sample: " ".join(sample['after_normalized']).strip(), project_samples))
    integrated_change = list(map(lambda sample: " ".join(sample['integrated_change']).strip(), project_samples))

    with open(os.path.join(out_dir, project_name + ".before_ctx"), "w") as f:
        f.write("\n".join(before_context))
    with open(os.path.join(out_dir, project_name + ".after_ctx"), "w") as f:
        f.write("\n".join(after_context))

    with open(os.path.join(out_dir, project_name + ".before_ctx_before"), "w") as f:
        f.write("\n".join(before_context_before))
    with open(os.path.join(out_dir, project_name + ".before_ctx_after"), "w") as f:
        f.write("\n".join(before_context_after))
    with open(os.path.join(out_dir, project_name + ".after_ctx_before"), "w") as f:
        f.write("\n".join(after_context_before))
    with open(os.path.join(out_dir, project_name + ".after_ctx_after"), "w") as f:
        f.write("\n".join(after_context_after))

    with open(os.path.join(out_dir, project_name + ".before_ctx_before_normalized"), "w") as f:
        f.write("\n".join(before_context_before_normalized))
    with open(os.path.join(out_dir, project_name + ".before_ctx_after_normalized"), "w") as f:
        f.write("\n".join(before_context_after_normalized))
    with open(os.path.join(out_dir, project_name + ".after_ctx_before_normalized"), "w") as f:
        f.write("\n".join(after_context_before_normalized))
    with open(os.path.join(out_dir, project_name + ".after_ctx_after_normalized"), "w") as f:
        f.write("\n".join(after_context_after_normalized))

    with open(os.path.join(out_dir, "before.txt"), "w") as f:
        f.write("\n".join(before))
    with open(os.path.join(out_dir, "after.txt"), "w") as f:
        f.write("\n".join(after))

    with open(os.path.join(out_dir, project_name + ".before_normalized"), "w") as f:
        f.write("\n".join(before_normalized))
    with open(os.path.join(out_dir, project_name + ".after_normalized"), "w") as f:
        f.write("\n".join(after_normalized))
    with open(os.path.join(out_dir, project_name + ".integrated_change"), "w") as f:
        f.write("\n".join(integrated_change))


def build_dataset(out_path, commits_path, inline_change_only=True, changed_context_only=True, merge_change_in_context=True, unique_change_only=True, non_trivial_change_only=False, min_changes=2, mov_in_change=False, overwrite_integarted_changes=False, context_size=10):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    print("Listing files...", file=sys.stderr)
    files = list(map(lambda p: (commits_path, p, inline_change_only, changed_context_only, merge_change_in_context, unique_change_only, non_trivial_change_only, min_changes, mov_in_change, overwrite_integarted_changes, context_size), os.listdir(commits_path)))
    print("Processing...", file=sys.stderr)
    project_samples_map = dict()
    sample_count = 0
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        with tqdm(total=len(files), dynamic_ncols=True) as pbar:
            for commits_samples, project_id in pool.imap(create_samples_task, files):
                pbar.update()
                if commits_samples is None:
                    continue
                sample_count += len(commits_samples)
                if project_id not in project_samples_map:
                    project_samples_map[project_id] = list()
                project_samples_map[project_id] += commits_samples
    print("Created {} samples".format(sample_count), file=sys.stderr)
    print("Pickling data...", file=sys.stderr)
    for k, v in tqdm(project_samples_map.items()):
        dump_samples(out_path, k, v)
    print("Done!", file=sys.stderr)


if __name__ == '__main__':
    commits_path = os.path.abspath(sys.argv[1])
    out_path = os.path.abspath(sys.argv[2])
    if not os.path.exists(out_path):
        os.mkdir(out_path)
        
    build_dataset(out_path, commits_path,
                  inline_change_only=False,
                  changed_context_only=True,
                  merge_change_in_context=True,
                  unique_change_only=False,
                  non_trivial_change_only=True,
                  min_changes=None,
                  mov_in_change=False,
                  overwrite_integarted_changes=False,
                  context_size=10)
