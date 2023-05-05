import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import logging


logger = logging.getLogger('VM.enc_dec_loader')


class TrainDataset(Dataset):
    def __init__(self, df, params):
        logger.info(f'Building training set...')
        self.vm_states, self.pm_states, self.target = df
        self.params = params
        self.train_len = self.vm_states.shape[0]
        logger.info(f'Train set size: {self.train_len}')

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        return self.vm_states[index], self.pm_states[index], self.target[index]


class WeightedSampler:
    def __init__(self, v, replacement=True):
        offset = np.min(v)
        v = v - offset + 1e-8
        self.weights = torch.as_tensor(np.abs(v[:, 0]) / np.sum(np.abs(v[:, 0])), dtype=torch.double)
        self.num_samples = self.weights.shape[0]
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples
