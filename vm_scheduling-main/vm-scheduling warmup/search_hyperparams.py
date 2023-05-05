"""
Grid search on the list of hyperparameters.
Example:
python search_hyperparams.py --dataset elect_coarse --gpu-ids 1 2 3 4
"""

import argparse
import logging
import multiprocessing
import os
import sys
import utils
import shutil
from copy import copy
from itertools import product
from subprocess import check_call


logger = logging.getLogger('TS.searcher')

utils.set_logger('param_search.log')

PYTHON = sys.executable
gpu_ids: list
param_template: utils.Params
args: argparse.ArgumentParser
search_params: dict

model_list = utils.model_list()


def launch_training_job(search_range):
    """ Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        search_range: one combination of the params to search
    """

    search_range = search_range[0]
    params = {k: search_params[k][search_range[idx]] for idx, k in enumerate(sorted(search_params.keys()))}
    model_param_list = '-'.join('_'.join((k, f'{v:.2f}')) for k, v in params.items())
    model_param = copy(param_template)
    for k, v in params.items():
        setattr(model_param, k, v)

    pool_id, job_idx = multiprocessing.Process()._identity
    gpu_id = gpu_ids[pool_id - 1]

    logger.info(f'Worker {pool_id} running {job_idx} using GPU {gpu_id}')

    # Create a new folder in parent_dir with unique_name 'job_name'
    model_name = os.path.join(model_dir, model_param_list)
    print('model name: ', model_name)
    if os.path.isdir(model_name):
        shutil.rmtree(model_name)
    os.makedirs(model_name)

    # Write parameters in json file
    json_path = os.path.join(model_name, 'params.json')
    model_param.save(json_path)
    logger.info(f'Params saved to: {json_path}')

    # Launch training with this config
    cmd = f'{PYTHON} train.py ' \
        f'--model-dir={args.model_dir} ' \
        f'--model={args.model} ' \
        f'--param-set={model_param_list} ' \
        f'--dataset={args.dataset} ' \
        f'--dataloader={args.dataloader} '

    logger.info(cmd)
    check_call(cmd, shell=True, env={'CUDA_VISIBLE_DEVICES': str(gpu_id),
                                     'OMP_NUM_THREADS': '4'})


def start_pool(project_list, processes):

    pool = multiprocessing.Pool(processes)
    pool.map(launch_training_job, [(i, ) for i in project_list])


def main():
    # Load the 'reference' parameters from parent_dir json file
    global param_template, gpu_ids, args, search_params, model_dir

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='elect_coarse', help='Name of the dataset')
    parser.add_argument('--dataloader', default='enc_dec', help='Which data loader to use')
    parser.add_argument('--model', default='enc_dec_relative', choices=model_list, help='Which model to use')
    parser.add_argument('--model-dir', default='param_search', help='Parent directory for all jobs')
    parser.add_argument('--gpu-ids', nargs='+', default=[0], type=int, help='GPU ids')
    args = parser.parse_args()

    # Load the parameters from json file
    data_dir = os.path.join('data', args.dataset)
    data_json_path = os.path.join(data_dir, 'params.json')
    assert os.path.isfile(data_json_path), f'No dataloader json configuration file found at {data_json_path}'
    param_template = utils.Params(data_json_path)

    model_dir = os.path.join('experiments', args.model_dir, args.dataset, args.model)
    json_file = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_file), f'No json configuration file found at {json_file}'
    param_template.update(json_file)

    gpu_ids = args.gpu_ids
    logger.info(f'Running on GPU: {gpu_ids}')

    # Perform hypersearch over parameters listed below
    search_params = {
        "embed_dim": [10, 20, 30],
    }

    keys = sorted(search_params.keys())
    search_range = list(product(*[[*range(len(search_params[i]))] for i in keys]))

    start_pool(search_range, len(gpu_ids))


if __name__ == '__main__':
    main()