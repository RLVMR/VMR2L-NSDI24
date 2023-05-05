import logging
import torch
import torch.nn as nn
from .components.pt_transformer import Transformer

logger = logging.getLogger('VM.pt_trans')


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.device = params.device
        self.batch_size = params.batch_size
        self.seq_length = params.num_pm

        self.pm_encode = nn.Linear(params.pm_cov, params.d_hidden)
        self.vm_encode = nn.Linear(params.vm_cov, params.d_hidden)
        self.pos_encoder = nn.Embedding(self.seq_length, params.d_hidden)

        self.transformer = Transformer(d_model=params.d_hidden, nhead=params.num_head,
                                       num_encoder_layers=params.transformer_blocks,
                                       num_decoder_layers=params.transformer_blocks, dim_feedforward=params.d_ff,
                                       activation='gelu', batch_first=True, dropout=params.dropout,
                                       need_attn_weights=True, device=params.device)

        if params.output_categorical:
            self.quantiles = params.quantiles
            self.output_layer = nn.Linear(params.d_hidden, len(self.quantiles) - 1)
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.output_layer = nn.Linear(params.d_hidden, 1)
            self.loss_fn = nn.L1Loss()
        self.output_categorical = params.output_categorical

    def forward(self, vm_states, pm_states, return_attns=False):
        vm_encode = self.vm_encode(vm_states)
        pm_encode = self.pm_encode(pm_states)
        pos_encode = self.pos_encoder(torch.arange(self.seq_length, device=self.device).unsqueeze(0))
        transformer_output = self.transformer(src=pm_encode + pos_encode, tgt=vm_encode)
        score = self.output_layer(transformer_output[0][:, 0])
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
        if self.output_categorical:
            logits, attn = self(vm_states, pm_states, return_attns=True)
            predict = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            predict, attn = self(vm_states, pm_states, return_attns=True)
        return {
            'predictions': predict,
            'enc_dec_attn_l0': attn[0],
            'enc_dec_attn_l1': attn[1]
        }
