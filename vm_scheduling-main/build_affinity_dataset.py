import argparse
import numpy as np
import os
import random
from tqdm import trange

from gym_reschdule_combination.envs.vm_rescheduler_env import parse_input

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-conflicts", type=int, default=500, help="number of conflicts per VM")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    num_conflicts = args.num_conflicts
    num_files = {
        'train': [0, 4000],
        'dev': [6000, 6200],
        'test': [6200, 6400]
    }
    for mode in ['train', 'dev', 'test']:
        print(f'Working on mode {mode}...')
        for env_id in trange(num_files[mode][0], num_files[mode][1], 1):
            scheduler = parse_input(f"./data/flex_vm_dataset/M/{mode}/flex_vm_{env_id}.json")
            pms = scheduler.active_pms
            vms = scheduler.migratable_vms
            num_vms = len(vms)
            all_vms = set(range(num_vms))
            # add index to all active pms.
            for i in range(len(pms)):
                pms[i].index = i

            # add index to all migratable vms.
            for i in range(num_vms):
                vms[i].index = i

            conflicts = np.zeros((num_vms, num_conflicts))
            for pm in pms:
                deployed_vm_idx = set()
                for vm in pm.vms.values():
                    deployed_vm_idx.add(vm.index)
                possible_conflicts = all_vms - deployed_vm_idx  # conflicts for all VMs on current PM

                for vm in pm.vms.values():
                    conflicts[vm.index] = random.sample(possible_conflicts, num_conflicts)

            np.save(f"./data/flex_vm_dataset/M/{mode}/{num_conflicts}_conflict_{env_id}.npy", conflicts)
