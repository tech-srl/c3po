import torch
import torch.utils.data as data
# Hack for solving OS Error while using multiprocessing data loading
# torch.multiprocessing.set_sharing_strategy('file_system')
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import Constants


class Dataset(data.Dataset):
    def __init__(self, split_path, device, ctx_vocab, args):
        self.samples = torch.load(split_path)
        self.size = len(self.samples)
        self.device = device
        self.context_mode = args.context_mode
        self.ctx_bos = ctx_vocab.getIndex(Constants.BOS_WORD)
        self.ctx_eos = ctx_vocab.getIndex(Constants.EOS_WORD)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        sample = self.samples[index]
        src = torch.LongTensor(sample['src'])
        tgt = torch.LongTensor(sample['tgt'])
        original_tgt = sample['original_tgt']
        if self.context_mode == 'full':
            before_ctx = torch.LongTensor(sample['before_ctx'])
            after_ctx = torch.LongTensor(sample['after_ctx'])
        elif self.context_mode == 'before':
            before_ctx = torch.LongTensor(sample['before_ctx_before'])
            after_ctx = torch.LongTensor(sample['after_ctx_before'])
        elif self.context_mode == 'after':
            before_ctx = torch.LongTensor(sample['before_ctx_after'])
            after_ctx = torch.LongTensor(sample['after_ctx_after'])
        elif self.context_mode == 'changes':
            before_ctx = torch.LongTensor(sample['before_ctx_changes'])
            after_ctx = torch.LongTensor(sample['after_ctx_changes'])
        elif self.context_mode == 'none':
            before_ctx = torch.LongTensor([self.ctx_bos, self.ctx_eos])
            after_ctx = torch.LongTensor([self.ctx_bos, self.ctx_eos])
        elif self.context_mode == 'path':
            before_ctx = torch.LongTensor(sample['before_ctx_path'])
            after_ctx = torch.LongTensor(sample['before_ctx_path'])
        else:
            raise Exception
        return src, tgt, before_ctx, after_ctx, original_tgt


def collate_fn(samples_list):
    srcs, tgts, before_ctx, after_ctx, original_tgts = zip(*samples_list)
    original_tgts = list(original_tgts)
    # (longest_seq, batch_size)
    packed_srcs = pack_sequence(srcs, enforce_sorted=False)
    packed_tgts = pack_sequence(tgts, enforce_sorted=False)
    padded_srcs, _ = pad_packed_sequence(packed_srcs, batch_first=True)
    padded_tgts, _ = pad_packed_sequence(packed_tgts, batch_first=True)
    mask = (padded_tgts != Constants.PAD).float()
    packed_before_ctx = pack_sequence(before_ctx, enforce_sorted=False)
    packed_after_ctx = pack_sequence(after_ctx, enforce_sorted=False)
    return padded_srcs, padded_tgts, mask, packed_before_ctx, packed_after_ctx, original_tgts

