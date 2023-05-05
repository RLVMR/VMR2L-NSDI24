import logging
import os
import torch
import torch.nn as nn
from .components.pt_transformer import TransformerUltraSparseDecoder, TransformerUltraSparseDecoderLayer

logger = logging.getLogger('VM.attn')


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.device = params.device
        self.num_pm = params.num_pm
        self.num_vm = params.num_vm
        self.seq_len = self.num_pm + self.num_vm + 2
        self.num_head = params.num_head

        self.d_hidden = params.d_hidden

        self.pm_encode = nn.Linear(params.pm_cov, params.d_hidden)
        self.vm_encode = nn.Linear(6, params.d_hidden)

        decoder_layer = TransformerUltraSparseDecoderLayer(d_model=params.d_hidden, nhead=params.num_head,
                                                           split_point=self.num_pm + 1,
                                                           dim_feedforward=params.d_ff, dropout=params.dropout,
                                                           activation='gelu', batch_first=True, norm_first=True,
                                                           need_attn_weights=True, device=params.device)
        self.transformer = TransformerUltraSparseDecoder(decoder_layer=decoder_layer,
                                                         num_layers=params.transformer_blocks)

        self.output_layer = nn.Linear(params.d_hidden, 1)
        self.critic_layer = nn.Linear(params.d_hidden, 1)
        self.critic_token = -torch.ones(1, 1, params.d_hidden).to(self.device)

        self.loss_fn = nn.L1Loss()

        self.vm_pm_relation = torch.unsqueeze(torch.arange(params.num_pm, device=self.device), dim=0)
        self.vm_pm_relation = torch.unsqueeze(self.vm_pm_relation, dim=-1)

    def forward(self, vm_states, num_step_states, pm_states, return_attns=False):
        b_sz = vm_states.shape[0]
        vm_pm_relation = self.vm_pm_relation.detach().clone().repeat(b_sz, 1, 1)
        vm_pm_relation = torch.cat([vm_pm_relation, vm_states[:, :, 6:7]], dim=1)
        vm_states = vm_states[:, :, :6]
        local_mask = torch.ones(b_sz, self.seq_len, self.seq_len, dtype=torch.bool, device=self.device)
        local_mask[:, 0, 0] = 0
        local_mask[:, -1, -1] = 0
        local_mask[:, 1:-1, 1:-1] = vm_pm_relation != vm_pm_relation[:, None, :, 0]
        local_mask = local_mask.view(b_sz, 1, self.seq_len, self.seq_len). \
            expand(-1, self.num_head, -1, -1).reshape(b_sz * self.num_head, self.seq_len, self.seq_len)
        tgt_key_pad_mask = torch.zeros(b_sz, self.seq_len, dtype=torch.bool, device=self.device)
        transformer_output = self.transformer(src=torch.cat([num_step_states.repeat(1, 1, self.d_hidden),
                                                             self.pm_encode(pm_states)], dim=1),
                                              tgt=torch.cat([self.vm_encode(vm_states),
                                                             self.critic_token.repeat(b_sz, 1, 1).detach()], dim=1),
                                              local_mask=local_mask, tgt_key_padding_mask=tgt_key_pad_mask)
        score = torch.squeeze(self.output_layer(transformer_output[1][:, :-1]))
        if return_attns:
            return score, transformer_output[2]
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
        b_sz = vm_states.shape[0]
        num_steps = torch.zeros(b_sz, 1, 1, device=vm_states.device)
        predict = self(vm_states, num_steps, pm_states)
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
        b_sz = vm_states.shape[0]
        num_steps = torch.zeros(b_sz, 1, 1, device=vm_states.device)
        predict = self(vm_states, num_steps, pm_states)
        return {
            'predictions': predict
        }
