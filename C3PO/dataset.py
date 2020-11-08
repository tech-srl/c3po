import os
import torch
import torch.utils.data as data
# Hack for solving OS Error while using multiprocessing data loading
# torch.multiprocessing.set_sharing_strategy('file_system')
from torch.nn.utils.rnn import pack_sequence
from functools import reduce
import numpy as np
import Constants


class Dataset(data.Dataset):
    def __init__(self, dir_path, device, args):
        self.samples = [os.path.join(dir_path, file)for file in os.listdir(dir_path)]
        self.size = len(self.samples)
        self.device = device
        self.context_mode = args.context_mode
        self.shuffle_paths = args.shuffle_path

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        sample = torch.load(self.samples[index])
        perm = None
        if self.shuffle_paths:
            perm = np.random.permutation(len(sample['paths'])).tolist()
        paths = Dataset.to_long_tensor_list(sample['paths'], perm)
        paths_positions = Dataset.to_long_tensor_list(sample['paths_positions'], perm)
        srcs = Dataset.to_long_tensor_list(sample['srcs'], perm)
        srcs_positions = Dataset.to_long_tensor_list(sample['srcs_positions'], perm)
        tgts = Dataset.to_long_tensor_list(sample['tgts'], perm)
        tgts_positions = Dataset.to_long_tensor_list(sample['tgts_positions'], perm)

        if self.context_mode == 'txt':
            ctx_txt = torch.LongTensor(sample['ctx_txt'])

            before_ctx_paths = Dataset.to_long_tensor_list([])
            before_ctx_paths_positions = Dataset.to_long_tensor_list([])
            before_ctx_srcs = Dataset.to_long_tensor_list([])
            before_ctx_srcs_positions = Dataset.to_long_tensor_list([])
            before_ctx_tgts = Dataset.to_long_tensor_list([])
            before_ctx_tgts_positions = Dataset.to_long_tensor_list([])

            after_ctx_paths = Dataset.to_long_tensor_list([])
            after_ctx_paths_positions = Dataset.to_long_tensor_list([])
            after_ctx_srcs = Dataset.to_long_tensor_list([])
            after_ctx_srcs_positions = Dataset.to_long_tensor_list([])
            after_ctx_tgts = Dataset.to_long_tensor_list([])
            after_ctx_tgts_positions = Dataset.to_long_tensor_list([])
        else:
            ctx_txt = torch.LongTensor([])
            suffix = 'none_' if self.context_mode == 'none' else ''
            before_ctx_paths = Dataset.to_long_tensor_list(sample[suffix + 'before_ctx_paths'])
            before_ctx_paths_positions = Dataset.to_long_tensor_list(sample[suffix + 'before_ctx_paths_positions'])
            before_ctx_srcs = Dataset.to_long_tensor_list(sample[suffix + 'before_ctx_srcs'])
            before_ctx_srcs_positions = Dataset.to_long_tensor_list(sample[suffix + 'before_ctx_srcs_positions'])
            before_ctx_tgts = Dataset.to_long_tensor_list(sample[suffix + 'before_ctx_tgts'])
            before_ctx_tgts_positions = Dataset.to_long_tensor_list(sample[suffix + 'before_ctx_tgts_positions'])

            after_ctx_paths = Dataset.to_long_tensor_list(sample[suffix + 'after_ctx_paths'])
            after_ctx_paths_positions = Dataset.to_long_tensor_list(sample[suffix + 'after_ctx_paths_positions'])
            after_ctx_srcs = Dataset.to_long_tensor_list(sample[suffix + 'after_ctx_srcs'])
            after_ctx_srcs_positions = Dataset.to_long_tensor_list(sample[suffix + 'after_ctx_srcs_positions'])
            after_ctx_tgts = Dataset.to_long_tensor_list(sample[suffix + 'after_ctx_tgts'])
            after_ctx_tgts_positions = Dataset.to_long_tensor_list(sample[suffix + 'after_ctx_tgts_positions'])


        if perm is not None:
            num_of_paths = len(sample['paths'])
            new_label = list()
            for t in sample['label']:
                if t == Constants.EOS:
                    new_label.append(t)
                else:
                    mov_op = t < num_of_paths + 2
                    if mov_op:
                        new_idx = perm.index(t - 2)
                        new_label.append(new_idx + 2)
                    else:
                        new_idx = perm.index(t - num_of_paths - 2)
                        new_label.append(new_idx + num_of_paths + 2)
            sample['label'] = new_label
        label = torch.LongTensor(sample['label'])
        paths_ops = sample['paths_ops']
        id_ = sample['id']

        return paths, paths_positions, srcs, srcs_positions, tgts, tgts_positions, \
               before_ctx_paths, before_ctx_paths_positions, before_ctx_srcs, before_ctx_srcs_positions, before_ctx_tgts, before_ctx_tgts_positions, \
               after_ctx_paths, after_ctx_paths_positions, after_ctx_srcs, after_ctx_srcs_positions, after_ctx_tgts, after_ctx_tgts_positions, \
               paths_ops, id_, label, ctx_txt

    @staticmethod
    def to_long_tensor_list(l, perm=None):
        if perm is not None:
            l = [l[i] for i in perm]
        return list(map(lambda x: torch.LongTensor(x), l))


def prepare_batch_aux(paths, paths_positions, srcs, srcs_positions, tgts, tgts_positions):
    num_of_paths = [len(p) for p in paths]
    paths = reduce(lambda a, b: a+b, paths)
    paths_positions = reduce(lambda a, b: a+b, paths_positions)
    srcs = reduce(lambda a, b: a+b, srcs)
    srcs_positions = reduce(lambda a, b: a+b, srcs_positions)
    tgts = reduce(lambda a, b: a+b, tgts)
    tgts_positions = reduce(lambda a, b: a+b, tgts_positions)
    # (longest_seq, batch_size)
    return {
        'paths': paths,
        'paths_positions': paths_positions,
        'srcs': srcs,
        'srcs_positions': srcs_positions,
        'tgts': tgts,
        'tgts_positions': tgts_positions,
        'num_of_paths': num_of_paths
    }


def pack_all(focus_dict, before_ctx_dict, after_ctx_dict):
        paths = focus_dict['paths'] + before_ctx_dict['paths'] + after_ctx_dict['paths']
        paths_positions = focus_dict['paths_positions'] + before_ctx_dict['paths_positions'] + after_ctx_dict['paths_positions']
        srcs = focus_dict['srcs'] + before_ctx_dict['srcs'] + after_ctx_dict['srcs']
        srcs_positions = focus_dict['srcs_positions'] + before_ctx_dict['srcs_positions'] + after_ctx_dict['srcs_positions']
        tgts = focus_dict['tgts'] + before_ctx_dict['tgts'] + after_ctx_dict['tgts']
        tgts_positions = focus_dict['tgts_positions'] + before_ctx_dict['tgts_positions'] + after_ctx_dict['tgts_positions']

        packed_paths = pack_sequence(paths, enforce_sorted=False)
        packed_paths_positions = pack_sequence(paths_positions, enforce_sorted=False)
        packed_srcs = pack_sequence(srcs, enforce_sorted=False)
        packed_srcs_positions = pack_sequence(srcs_positions, enforce_sorted=False)
        packed_tgts = pack_sequence(tgts, enforce_sorted=False)
        packed_tgts_positions = pack_sequence(tgts_positions, enforce_sorted=False)
        res = {
            'packed_paths': packed_paths,
            'packed_paths_positions': packed_paths_positions,
            'packed_srcs': packed_srcs,
            'packed_srcs_positions': packed_srcs_positions,
            'packed_tgts': packed_tgts,
            'packed_tgts_positions': packed_tgts_positions,
            'focus_num_of_paths': focus_dict['num_of_paths'],
            'before_ctx_num_of_paths': before_ctx_dict['num_of_paths'],
            'after_ctx_num_of_paths': after_ctx_dict['num_of_paths']
            }
        return res


def collate_fn(samples_list):
    paths, paths_positions, srcs, srcs_positions, tgts, tgts_positions, \
    before_ctx_paths, before_ctx_paths_positions, before_ctx_srcs, before_ctx_srcs_positions, before_ctx_tgts, before_ctx_tgts_positions, \
    after_ctx_paths, after_ctx_paths_positions, after_ctx_srcs, after_ctx_srcs_positions, after_ctx_tgts, after_ctx_tgts_positions, \
    path_ops, id_, label, ctx_txt = zip(*samples_list)
    path_ops = list(path_ops)
    id_ = list(id_)
    focus_dict = prepare_batch_aux(paths, paths_positions, srcs, srcs_positions, tgts, tgts_positions)
    before_ctx_dict = prepare_batch_aux(before_ctx_paths, before_ctx_paths_positions, before_ctx_srcs, before_ctx_srcs_positions, before_ctx_tgts, before_ctx_tgts_positions)
    after_ctx_dict = prepare_batch_aux(after_ctx_paths, after_ctx_paths_positions, after_ctx_srcs, after_ctx_srcs_positions, after_ctx_tgts, after_ctx_tgts_positions)
    samples_dict = pack_all(focus_dict, before_ctx_dict, after_ctx_dict)
    if len(ctx_txt[0].tolist()) != 0:
        samples_dict['packed_ctx'] = pack_sequence(ctx_txt, enforce_sorted=False)
        del samples_dict['before_ctx_num_of_paths']
        del samples_dict['after_ctx_num_of_paths']
    packed_label = pack_sequence(label, enforce_sorted=False)
    return samples_dict, packed_label, path_ops, id_