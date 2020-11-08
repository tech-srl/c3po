import torch.nn as nn
from utils import *
from torch.nn.utils.rnn import  pad_packed_sequence, pack_sequence, pack_padded_sequence
import Constants
try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


class Trainer(object):
    def __init__(self, args, encoder, decoder, optimizer, criterion, device, disable_prog_bar):
        super(Trainer, self).__init__()
        self.args = args
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion
        self.epoch = 0
        self.softmax = nn.Softmax(dim=1)
        self.disable_prog_bar = disable_prog_bar
        self.non_blocking = args.pin_memory
        self.use_attention = args.attention
        self.accum_grad = args.accum_grad
        self.use_amp = APEX_AVAILABLE

    @staticmethod
    def create_inputs(packed_encoded_path, packed_label, init_token, remove_eos=True):
        '''
        :param packed_encoded_path: PackedSequence
        :param packed_label: PackedSequence
        :param init_token: of shape (1, h_dim)
        :param encoded_ctx: of shape (batch, h_dim)
        :param remove_eos: boolean
        :return: labels_list: PackedSequence
        '''
        # encoded_path: (batch_size, length_of_longest_seq, h_dim)
        encoded_path, encoded_path_lengths = pad_packed_sequence(packed_encoded_path, batch_first=True)
        if init_token is not None:
            encoded_path = encoded_path.type(init_token.type())
        batch_size, _, h_dim = encoded_path.size()
        labels_list = list()
        if packed_label is None:
            for i in range(batch_size):
                # (1 , h_dim)
                cur_labels = init_token
                labels_list.append(cur_labels)
        else:
            remove = 1 if remove_eos is True else 0
            # labels: (batch_size, length_of_longest_seq, 1)
            labels, labels_lengths = pad_packed_sequence(packed_label, batch_first=True)
            for i in range(batch_size):
                # (labels, h_dim)
                cur_labels = encoded_path[i, labels[i][:labels_lengths[i]-remove]] # -1 is for removing EOS token from input
                if init_token is not None:
                    # (1 + labels, h_dim)
                    cur_labels = torch.cat([init_token, cur_labels], dim=0)
                labels_list.append(cur_labels)
        return pack_sequence(labels_list, enforce_sorted=False)

    def calculate_acc_and_loss(self, padded_label, padded_weights, res, batch_predictions=None):
        '''
        :param padded_label: of shape (batch_size, output_len)
        :param padded_weights: of shape  (batch_size, encoded_outputs_len, output_len)
        :param batch_predictions: of shape (batch_size, self.args.max_seq_len + 1)
        :return:
        '''
        batch_size, output_len = padded_label.shape
        # (batch_size * output_len)
        padded_label_flat = padded_label.view(-1)
        # (batch_size, output_len, encoded_outputs_len)
        padded_weights = padded_weights.transpose(1, 2).contiguous()
        # (batch_size * output_len, encoded_outputs_len)
        padded_weights_flat = padded_weights.reshape(batch_size * output_len, -1).contiguous()
        # (batch_size * output_len)
        pad_mask = padded_label_flat != Constants.PAD
        if batch_predictions is None:
            # (batch_size * out_len, encoded_outputs_len)
            scores = self.softmax(padded_weights_flat).cpu()
            # with torch.backends.cudnn.benchmark = True, topk on cuda tensor gives indecies out of range when there is NaN
            # for example: tensor = [[0.0001],[0.0001],[nan]]
            # (batch_size * out_len, 1)
            _, predicted_labels = scores.topk(1, dim=1)
            predicted_labels = predicted_labels.to(self.device)
            full_predicted_labels = predicted_labels.reshape(batch_size, output_len)
        else:
            full_predicted_labels = batch_predictions
            # (batch_size, output_len)
            predicted_labels = torch.zeros(size=(batch_size, output_len), device=self.device).long()
            predicted_labels[:, :batch_predictions.shape[1]] = batch_predictions[:, :output_len]
        res['predictions'] += full_predicted_labels.tolist()
        res['targets'] += padded_label.tolist()
        # (batch_size * output_len)
        predicted_labels = predicted_labels.reshape((batch_size * output_len))
        predicted_labels = predicted_labels[pad_mask]
        true_labels_masked = padded_label_flat[pad_mask]
        total_correct = (true_labels_masked == predicted_labels).sum().item()
        total_targets = len(true_labels_masked)

        loss = self.criterion(padded_weights_flat, padded_label_flat)
        return total_correct, total_targets, loss

    @staticmethod
    def calculate_metrics(res):
        predictions = res['predictions']
        targets = res['targets']
        path_ops = res['path_ops']
        res['predicted_ops'] = list()

        total_samples = len(predictions)
        total_correct = 0

        true_positive = 0
        false_positive = 0
        false_negative = 0

        precision = 0
        recall = 0
        f1 = 0
        for i in range(total_samples):
            num_of_paths = len(path_ops[i])
            pred_tokens = list(filter(lambda t: t != Constants.PAD and t != Constants.EOS, predictions[i]))
            r = list()
            for t in pred_tokens:
                if t < num_of_paths + Constants.NUM_OF_CTRL_TOKENS:
                    r.append(path_ops[i][t-2])
                elif t < 2 * num_of_paths + Constants.NUM_OF_CTRL_TOKENS:
                    r.append(path_ops[i][t - num_of_paths - Constants.NUM_OF_CTRL_TOKENS].replace("MOV", "UPD"))
                else:
                    r.append(path_ops[i][t - 2 * num_of_paths - Constants.NUM_OF_CTRL_TOKENS].replace("MOV", "INS"))
            res['predicted_ops'].append(r)
            target_tokens = list(filter(lambda t: t != Constants.PAD and t != Constants.EOS, targets[i]))
            total_correct += 1 if pred_tokens == target_tokens else 0
            for token in pred_tokens:
                if token in target_tokens:
                    true_positive += 1
                else:
                    false_positive += 1
            for token in target_tokens:
                if token not in pred_tokens:
                    false_negative += 1

        if true_positive + false_positive != 0:
            precision = true_positive / (true_positive + false_positive)
        if true_positive + false_negative != 0:
            recall = true_positive / (true_positive + false_negative)
        if (precision + recall) != 0:
            f1 = 2 * precision * recall / (precision + recall)
        res['precision'] = precision
        res['recall'] = recall
        res['f1'] = f1
        res['acc'] = total_correct / total_samples
        del res['path_ops']

    def packed_to_deivce(self, entry):
        entry['packed_paths'] = entry['packed_paths'].to(device=self.device, non_blocking=self.non_blocking)
        entry['packed_paths_positions'] = entry['packed_paths_positions'].to(device=self.device, non_blocking=self.non_blocking)
        entry['packed_srcs'] = entry['packed_srcs'].to(device=self.device, non_blocking=self.non_blocking)
        entry['packed_srcs_positions'] = entry['packed_srcs_positions'].to(device=self.device, non_blocking=self.non_blocking)
        entry['packed_tgts'] = entry['packed_tgts'].to(device=self.device, non_blocking=self.non_blocking)
        entry['packed_tgts_positions'] = entry['packed_tgts_positions'].to(device=self.device, non_blocking=self.non_blocking)
        if 'packed_ctx' in entry.keys():
            entry['packed_ctx'] = entry['packed_ctx'].to(device=self.device, non_blocking=self.non_blocking)

    def train(self, data_generator):
        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        num_of_batches = 0
        total_targets = 0
        total_correct = 0
        res = {'predictions': list(), 'targets': list(), 'ids': list(), 'path_ops': list()}
        pbar = tqdm(data_generator, desc='Training epoch ' + str(self.epoch + 1) + '', disable=self.disable_prog_bar, dynamic_ncols=True)
        for i, (samples_dict, packed_label, path_ops, ids) in enumerate(pbar):
            self.packed_to_deivce(samples_dict)
            packed_label = packed_label.to(device=self.device, non_blocking=self.non_blocking)
            num_of_paths = samples_dict['focus_num_of_paths']
            batch_size = len(num_of_paths)
            res['ids'] += ids
            res['path_ops'] += path_ops
            # encoded_path: (batch_1, h_dim)
            # encoded_ctx: (batch, num_of_paths, 2 * h_dim)
            packed_encoded_path, packed_encoded_ctx, h = self.encoder(**samples_dict)
            inputs = self.create_inputs(packed_encoded_path, packed_label, self.decoder.get_init_token())
            c = self.decoder.create_h_or_c(batch_size)
            hc = (h, c)
            batch_size = len(num_of_paths)
            # attention_weights: PadSequence of shape (batch_size, output_len, query_len)
            packed_attention_scores, hc = self.decoder(packed_encoded_path, packed_encoded_ctx, batch_size, inputs, hc)
            # (batch_size, output_len)
            padded_label, labels_lengths = pad_packed_sequence(packed_label, batch_first=True)
            # (batch_size, encoded_outputs_len, output_len)
            padded_weights, weights_lengths = pad_packed_sequence(packed_attention_scores, batch_first=True, padding_value=Constants.NEG_INF)
            correct, total, loss = self.calculate_acc_and_loss(padded_label, padded_weights, res)
            total_correct += correct
            total_targets += total

            loss = loss / self.accum_grad
            if self.use_amp:
                if (i + 1) % self.accum_grad == 0:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    with amp.scale_loss(loss, self.optimizer, delay_unscale=True) as scaled_loss:
                        scaled_loss.backward()
                    nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 5)
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.encoder.parameters(), 5)
                nn.utils.clip_grad_norm_(self.decoder.parameters(), 5)
                if (i + 1) % self.accum_grad == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item()
            num_of_batches += 1
            pbar.set_postfix_str("Loss: {}".format(total_loss / num_of_batches))
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.calculate_metrics(res)
        loss = total_loss / num_of_batches
        self.epoch += 1
        return loss, res

    def test(self, data_generator):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            total_loss = 0.0
            num_of_batches = 0
            total_targets = 0
            total_correct = 0
            res = {'predictions': list(), 'targets': list(), 'ids': list(), 'path_ops': list()}
            pbar = tqdm(data_generator, desc='Testing epoch  ' + str(self.epoch) + '', disable=self.disable_prog_bar, dynamic_ncols=True)
            for samples_dict, packed_label, path_ops, ids in pbar:
                self.packed_to_deivce(samples_dict)
                packed_label = packed_label.to(device=self.device, non_blocking=self.non_blocking)
                num_of_paths = samples_dict['focus_num_of_paths']
                batch_size = len(num_of_paths)
                res['ids'] += ids
                res['path_ops'] += path_ops
                # encoded_path: (batch_1, h_dim)
                # encoded_ctx: (batch, h_dim)
                packed_encoded_path, packed_encoded_ctx, h = self.encoder(**samples_dict)
                inputs = self.create_inputs(packed_encoded_path, None, self.decoder.get_init_token())
                encoded_outputs_len = Constants.NUM_OF_CTRL_TOKENS + Constants.NUM_OF_OPS * max(num_of_paths)
                c = self.decoder.create_h_or_c(batch_size)
                hc = (h, c)
                batch_predictions = torch.zeros(size=(batch_size, self.args.max_seq_len + 1)).long()
                # (batch_size, output_len)
                padded_label, labels_lengths = pad_packed_sequence(packed_label, batch_first=True)
                # (batch_size, output_len, encoded_outputs_len)
                batch_scores = torch.zeros(size=(batch_size, padded_label.shape[1], encoded_outputs_len), device=self.device)
                # False (batch_size, 1)
                pad_mask = torch.zeros(size=(batch_size, 1), device=self.device) == 1
                for t in range(self.args.max_seq_len + 1):
                    # attention_weights: PadSequence of shape (batch_size, 1, query_len)
                    packed_attention_scores, hc = self.decoder(packed_encoded_path, packed_encoded_ctx, batch_size, inputs, hc)
                    # (batch_size, encoded_outputs_len, 1)
                    padded_weights, weights_lengths = pad_packed_sequence(packed_attention_scores, batch_first=True, padding_value=Constants.NEG_INF)
                    # (batch_size, 1, encoded_outputs_len)
                    padded_weights = padded_weights.transpose(1, 2).contiguous()
                    if t < batch_scores.shape[1]:
                        batch_scores[:, t] = padded_weights.squeeze(dim=1)
                    # (batch_size, 1)
                    predicted_labels = self.calculate_predicted_labels(padded_weights, batch_predictions[:, :t], force_non_repetition=True)
                    # (batch_size, 1)
                    predicted_labels[pad_mask] = Constants.PAD
                    batch_predictions[:, t] = predicted_labels.squeeze()
                    # (batch_size, 1)
                    ended = predicted_labels == Constants.EOS
                    pad_mask |= ended
                    predicted_labels_packed = pack_padded_sequence(predicted_labels, batch_size * [1], batch_first=True, enforce_sorted=False)
                    inputs = self.create_inputs(packed_encoded_path, predicted_labels_packed, None, remove_eos=False)
                # (batch_size, encoded_outputs_len, output_len)
                batch_scores = batch_scores.transpose(1, 2).contiguous()
                correct, total, loss = self.calculate_acc_and_loss(padded_label, batch_scores, res, batch_predictions)
                total_correct += correct
                total_targets += total
                total_loss += loss.item()
                num_of_batches += 1
                pbar.set_postfix_str("Loss: {}".format(total_loss / num_of_batches))
            self.calculate_metrics(res)
            loss = total_loss / num_of_batches
            return loss, res

    def calculate_predicted_labels(self, padded_weights, batch_predictions, force_non_repetition=True):
        # (batch_size, encoded_outputs_len)
        scores = self.softmax(padded_weights.squeeze(dim=1)).cpu()
        if force_non_repetition is False:
            # (batch_size, 1)
            _, predicted_labels = scores.topk(1, dim=1)
            predicted_labels = predicted_labels.to(self.device)
            return predicted_labels

        # (batch_size, max_seq_len)
        _, predicted_labels = scores.topk(self.args.max_seq_len+1, dim=1, sorted=True)
        predicted_labels = predicted_labels.to(self.device)
        batch_size = predicted_labels.shape[0]
        calc_predicted_labels = torch.zeros(size=(batch_size, 1), device=self.device).long()
        for i in range(batch_size):
            already_predicted = batch_predictions[i].tolist()
            calc_predicted_labels[i] = list(filter(lambda x: x not in already_predicted, predicted_labels[i].tolist()))[0]
        return calc_predicted_labels
