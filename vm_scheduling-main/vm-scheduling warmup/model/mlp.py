import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger('VM.mlp')


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.device = params.device
        self.batch_size = params.batch_size
        self.input_size = params.num_pm * params.pm_cov + params.vm_cov
        self.output_size = 1
        self.fc1 = nn.Linear(self.input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 32)
        if params.output_categorical:
            self.quantiles = params.quantiles
            self.fc5 = nn.Linear(32, len(self.quantiles) - 1)
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.fc5 = nn.Linear(32, self.output_size)
            self.loss_fn = nn.L1Loss()
        self.output_categorical = params.output_categorical

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

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
        predict = self(torch.cat([pm_states.reshape(vm_states.shape[0], -1),
                                  vm_states.reshape(vm_states.shape[0], -1)], dim=-1))
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
            logits = self(torch.cat([pm_states.reshape(vm_states.shape[0], -1),
                                     vm_states.reshape(vm_states.shape[0], -1)], dim=-1))
            predict = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            predict = self(torch.cat([pm_states.reshape(vm_states.shape[0], -1),
                                      vm_states.reshape(vm_states.shape[0], -1)], dim=-1))
        return {
            'predictions': predict,
        }
