import logging
import os
import torch
import torch.nn as nn
from .components.pt_transformer import Transformer


logger = logging.getLogger('VM.attn')


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.device = params.device
        # self.seq_length = params.num_pm
        self.d_hidden = params.d_hidden

        self.pm_encode = nn.Linear(params.pm_cov, params.d_hidden)
        self.vm_encode = nn.Linear(params.vm_cov, params.d_hidden)

        self.transformer = Transformer(d_model=params.d_hidden, nhead=params.num_head,
                                       num_encoder_layers=params.transformer_blocks,
                                       num_decoder_layers=params.transformer_blocks, dim_feedforward=params.d_ff,
                                       activation='gelu', batch_first=True, dropout=params.dropout,
                                       need_attn_weights=True, device=params.device)

        self.output_layer = nn.Linear(params.d_hidden, 1)
        self.loss_fn = nn.L1Loss()

    def forward(self, vm_states, pm_states, return_attns=False):
        vm_encode = self.vm_encode(vm_states)
        pm_encode = self.pm_encode(pm_states)
        transformer_output = self.transformer(src=pm_encode, tgt=vm_encode)
        score = torch.squeeze(self.output_layer(transformer_output[0]))
        if return_attns:
            return score, transformer_output[1]
        else:
            return score

    def do_train(self, vm_states, pm_states, labels):
        """ Train for one batch.
        Args:
            vm_states ([batch_size, num_vm, vm_cov]): features of virtual machines
            pm_states ([batch_size, num_pm, pm_cov]): features of physical machines
            labels ([batch_size, 1]): ground truth of target
        Returns:
            loss_list (list): list of each individual loss component for plotting
            loss (int): total loss for backpropagation
        """
        predict = self(vm_states, pm_states)
        loss = self.loss_fn(predict, labels)
        loss.backward()
        return [loss.item()], loss

    def test(self, vm_states, pm_states, labels):
        """ Test for one batch.
        Args:
            vm_states ([batch_size, num_vm, vm_cov]): features of virtual machines
            pm_states ([batch_size, num_pm, pm_cov]): features of physical machines
            labels ([batch_size, 1]): ground truth of target
        Returns:
            sampled_params ([batch_size, predict_steps, num_quantiles]): return quantiles as specified in params.json.
            enc_self_attention ([batch_size, predict_steps, predict_start]): to visualize encoder-decoder attention.
        """
        predict = self(vm_states, pm_states)
        return {
            'predictions': predict
        }
