import torch.nn as nn
from utils import *
from torch.nn.utils.rnn import  pad_packed_sequence
import Constants
try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


class Trainer(object):
    def __init__(self, args, tgt_vocab, model, optimizer, device, disable_prog_bar):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.tgt_vocab = tgt_vocab
        self.epoch = 0
        self.softmax = nn.Softmax(dim=1)
        self.disable_prog_bar = disable_prog_bar
        self.non_blocking = args.pin_memory
        self.accum_grad = args.accum_grad
        self.use_amp = APEX_AVAILABLE
        # self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    def train(self, data_generator):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        total_samples = 0
        num_of_batches = 0
        pbar = tqdm(data_generator, desc='Training epoch ' + str(self.epoch + 1) + '', disable=self.disable_prog_bar, dynamic_ncols=True)
        for i, (padded_srcs, padded_tgts, mask, packed_before_ctx, packed_after_ctx, original_tgts) in enumerate(pbar):
            padded_srcs = padded_srcs.to(device=self.device, non_blocking=self.non_blocking)
            padded_tgts = padded_tgts.to(device=self.device, non_blocking=self.non_blocking)
            packed_before_ctx = packed_before_ctx.to(device=self.device, non_blocking=self.non_blocking)
            packed_after_ctx = packed_after_ctx.to(device=self.device, non_blocking=self.non_blocking)
            mask = mask.to(device=self.device, non_blocking=self.non_blocking)
            loss = self.model.loss(padded_srcs, packed_before_ctx, packed_after_ctx, padded_tgts, mask)
            loss = loss / self.accum_grad
            if self.use_amp:
                if (i + 1) % self.accum_grad == 0:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    with amp.scale_loss(loss, self.optimizer, delay_unscale=True) as scaled_loss:
                        scaled_loss.backward()
            else:
                loss.backward()
                if (i + 1) % self.accum_grad == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            # loss.backward()
            total_samples += padded_srcs.shape[0]
            total_loss += loss.item()
            num_of_batches += 1
            # if (i + 1) % self.accum_grad == 0:
            #     self.optimizer.step()
            #     self.optimizer.zero_grad()
            # self.optimizer.step()
            # self.optimizer.zero_grad()
            pbar.set_postfix_str("Loss: {}".format(total_loss / total_samples))
        self.optimizer.step()
        self.optimizer.zero_grad()
        loss = total_loss / total_samples
        self.epoch += 1
        # self.scheduler.step()
        return loss

    def test(self, data_generator):
        self.model.eval()
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            num_of_batches = 0
            pbar = tqdm(data_generator, desc='Testing epoch  ' + str(self.epoch) + '', disable=self.disable_prog_bar, dynamic_ncols=True)
            for padded_srcs, padded_tgts, mask, packed_before_ctx, packed_after_ctx, original_tgts in pbar:
                padded_srcs = padded_srcs.to(device=self.device, non_blocking=self.non_blocking)
                packed_before_ctx = packed_before_ctx.to(device=self.device, non_blocking=self.non_blocking)
                packed_after_ctx = packed_after_ctx.to(device=self.device, non_blocking=self.non_blocking)
                mask = mask.to(device=self.device, non_blocking=self.non_blocking)
                scores, seqs = self.model(padded_srcs, packed_before_ctx, packed_after_ctx, mask)
                total_samples += padded_srcs.shape[0]
                for i, seq in enumerate(seqs):
                    str_seq = " ".join(map(lambda t: self.tgt_vocab.getLabel(t), seq))
                    total_correct += 1 if str_seq == original_tgts[i] else 0
                num_of_batches += 1
            acc = total_correct / total_samples
            return acc
