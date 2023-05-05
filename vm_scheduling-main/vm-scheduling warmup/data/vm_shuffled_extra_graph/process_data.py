import os
import numpy as np
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


logger = logging.getLogger('VM.process_data')


def data_split(params):
    df = pd.read_pickle(os.path.join(params.raw_data_path, params.name))
    target = np.float32(df.iloc[:, -params.num_vm:].values)
    vm_states = np.float32(df.iloc[:, -params.num_vm * (1 + params.vm_cov):-params.num_vm].values).reshape(-1, params.num_vm, params.vm_cov)
    pm_states = np.float32(df.iloc[:, :params.num_pm * params.pm_cov].values).reshape(-1, params.num_pm, params.pm_cov)
    num_val = int(df.shape[0] * params.val_portion)
    num_test = int(df.shape[0] * params.test_portion)
    num_train = df.shape[0] - num_val - num_test
    # scaler = StandardScaler().fit(vm_states[:num_train])
    # data = scaler.transform(data)
    np.save(os.path.join(params.raw_data_path, f'{params.target_save_name}_gt_val.npy'),
            target[num_train:num_train+num_val])
    np.save(os.path.join(params.raw_data_path, f'{params.target_save_name}_gt_test.npy'), target[-num_test:])
    return (vm_states[:num_train], pm_states[:num_train], target[:num_train]), \
           (vm_states[num_train:num_train+num_val], pm_states[num_train:num_train+num_val], target[num_train:num_train+num_val]),\
           (vm_states[-num_test:], pm_states[-num_test:], target[-num_test:])
