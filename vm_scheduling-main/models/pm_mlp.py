import logging
import os
import torch
import torch.nn as nn


logger = logging.getLogger('VM.mlp')


class PM_candidate_model(nn.Module):
    def __init__(self, params):
        super(PM_candidate_model, self).__init__()
        self.device = params.device
        self.input_size = params.num_pm * params.pm_cov + params.vm_cov  # 2246
        self.output_size = params.num_pm

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.pm_head = nn.Linear(128, self.output_size)

        self.loss_fn = nn.L1Loss()

    def forward(self, vm_states, pm_states):
        b_sz = vm_states.shape[0]
        x = torch.cat([pm_states.reshape(b_sz, -1), vm_states.reshape(b_sz, -1)], dim=-1)
        hidden = self.layers(x)
        x = self.pm_head(hidden)
        return x


class PM_MLP_Wrapper(nn.Module):
    def __init__(self, params):
        super(PM_MLP_Wrapper, self).__init__()
        self.model = PM_candidate_model(params).to(params.device)
