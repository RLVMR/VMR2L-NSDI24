import argparse
import copy
import logging
import os
import importlib
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import utils

matplotlib.use('Agg')
logger = logging.getLogger('VM.eval')
model_list = utils.model_list()


def evaluate(model, test_loader, params, plot_num, validation=False):
    """ Evaluate the model on the test set.
    Args:
        model (torch.nn.Module): the neural network model
        test_loader (Dataloader): load test data and labels
        params (Params): hyperparameters
        plot_num (int): if(-1) - evaluation from evaluate.py; else(epoch) - evaluation on epoch
        validation (boolean): using validation (True) or test (False)
    """
    model.eval()
    with torch.no_grad():
        if len(test_loader) == 1:
            plot_batch = 0
        else:
            plot_batch = np.random.randint(len(test_loader) - 1)

        num_windows = params.num_val_windows if validation else params.num_test_windows
        predict_all = torch.zeros(num_windows, params.num_vm, device=params.device)
        window_count = 0

        # Test_loader:
        # x ([batch_size, test_window, time_dim]): known inputs;
        # exo ([batch_size, predict_start, exo_dim]): exogenous inputs;
        # meta ([batch_size, meta_dim]): static info;
        # sampled ([batch_size, test_window]): randomly sampled index from scale window;
        # v ([batch_size, 2, 1]): scaling for each window in the batch;
        # label_plot ([batch_size, test_window]): only the last predict_steps are used for metrics;
        # plot_cov ([batch_size, test_window]): covariates used for plots.

        for i, batch in enumerate(tqdm(test_loader)):
            vm_states, pm_states, labels = map(lambda x: x.to(params.device), batch)
            batch_size = vm_states.shape[0]
            window_end = window_count + batch_size

            result = model.test(vm_states, pm_states, labels)
            predictions = result.get('predictions')
            if i == plot_batch:
                enc_dec_attn_l0 = result.get('enc_dec_attn_l0')  # Nullable
                enc_dec_attn_l1 = result.get('enc_dec_attn_l1')  # Nullable
                plot_predictions = copy.deepcopy(predictions.cpu().data.numpy())
                plot_labels = copy.deepcopy(labels.cpu().data.numpy())
            predict_all[window_count:window_end] = predictions
            window_count = window_end

        predict_all = predict_all.cpu().data.numpy()
        if not validation:
            np.save(os.path.join(params.model_dir, 'predictions.csv'), predict_all)
            gt_all = np.load(os.path.join(params.raw_data_path, f'{params.target_save_name}_gt_test.npy'))
        else:
            np.save(os.path.join(params.model_dir, 'predictions_val.csv'), predict_all)
            gt_all = np.load(os.path.join(params.raw_data_path, f'{params.target_save_name}_gt_val.npy'))

        summary_metric = utils.calc_metrics_all(predict_all, gt_all, params)
        metrics_string = '; '.join('{}: {:05.3f}'.format(k, v) for k, v in summary_metric.items())
        if not validation:
            metrics_string += '   <==='
        logger.info('- Full {} metrics: {}'.format('validation' if validation else 'test', metrics_string))

    if not validation:
        # plot_samples(params.plot_dir, predict_all, gt_all, f'{plot_num}_test', summary_metric, params,
        #              enc_dec_attn_l0, enc_dec_attn_l1)
        return summary_metric, predict_all
    else:
        # plot_samples(params.plot_dir, predict_all, gt_all, f'{plot_num}_val', summary_metric, params,
        #              enc_dec_attn_l0, enc_dec_attn_l1)
        return summary_metric

"""
def plot_samples(plot_dir,
                 predictions,
                 labels,
                 plot_num,
                 plot_metrics,
                 params,
                 enc_dec_attn_l0, enc_dec_attn_l1):
    f = plt.figure(figsize=(12, 12), constrained_layout=True)
    if params.output_categorical:
        num_class = len(params.quantiles) - 1
        cat_count = np.zeros((num_class, num_class))
        for cat in range(num_class):
            for pred in range(num_class):
                cat_count[cat, pred] = np.sum((predictions == pred) & (labels == cat))
            cat_count[cat] = cat_count[cat] / np.sum(cat_count[cat])
        ax = sns.heatmap(cat_count, linewidth=0.5)
    else:
        plt.scatter(predictions, labels)
    plot_metrics_str = f'Epoch {plot_num} - MAE: {plot_metrics["mae"]:.3f}, MSE: {plot_metrics["mse"]:.3f} '
    plt.title(plot_metrics_str, fontsize=10)
    plt.xlabel('predictions')
    plt.ylabel('ground truth')
    f.savefig(os.path.join(plot_dir, f'scatter_epoch_{plot_num}.png'))
    plt.close()

    if enc_dec_attn_l0 is not None:
        enc_dec_attn_l0 = enc_dec_attn_l0.cpu().data.numpy()
        enc_dec_attn_l1 = enc_dec_attn_l1.cpu().data.numpy()
        f = plt.figure(figsize=(12, 42), constrained_layout=True)
        nrows = 10
        ncols = 2

        ax = f.subplots(nrows, ncols)
        random_index = np.random.randint(enc_dec_attn_l0.shape[0], size=nrows)
        for i, idx in enumerate(random_index):
            ax[i, 0].set_title('Layer 0 Enc-dec attn', fontsize=10)
            ax[i, 0].hist(enc_dec_attn_l0[idx, 0], bins=30)
            ax[i, 1].set_title('Layer 1 Enc-dec attn', fontsize=10)
            ax[i, 1].hist(enc_dec_attn_l1[idx, 0], bins=30)
        f.savefig(os.path.join(plot_dir, f'attn_epoch_{plot_num}.png'))
        plt.close()
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='vm_scheduler', help='Name of the dataset')
    parser.add_argument('--dataloader', default='enc_dec_attn', help='Which data loader to use')
    parser.add_argument('--model', default='enc_dec_attn', choices=model_list, help='Which model to use')
    parser.add_argument('--model-dir', default='base_model',
                        help='Directory containing params.json, and training results')
    parser.add_argument('--param-set', default=None, help='Set of model parameters created for hypersearch')
    parser.add_argument('--restore-file', default='best',
                        help='Required, name of the file in --model_dir containing weights to evaluate')

    args = parser.parse_args()

    # Load the parameters from json file
    data_dir = os.path.join('data', args.dataset)
    data_json_path = os.path.join(data_dir, 'params.json')
    assert os.path.isfile(data_json_path), f'No dataloader json configuration file found at {data_json_path}'
    params = utils.Params(data_json_path)

    if args.param_set is not None:
        params.model_dir = os.path.join('experiments', args.model_dir, args.dataset, args.model,
                                        args.dataloader, args.param_set)
    else:
        params.model_dir = os.path.join('experiments', args.model_dir, args.dataset, args.model, args.dataloader)
    json_path = os.path.join(params.model_dir, 'params.json')
    assert os.path.isfile(json_path), f'No model json configuration file found at {json_path}'
    params.update(json_path)

    utils.set_logger(os.path.join(params.model_dir, 'eval.log'))
    logger = logging.getLogger(f'VM.{args.model}')

    params.dataset = args.dataset
    params.model = args.model
    params.plot_dir = os.path.join(params.model_dir, 'figures')
    if args.param_set is not None:
        params.plot_title = os.path.join(args.model_dir, args.dataset, args.model, args.dataloader,
                                         args.param_set)
    else:
        params.plot_title = os.path.join(args.model_dir, args.dataset, args.model, args.dataloader)

    net = importlib.import_module(f'model.{args.model}')
    data_loader = importlib.import_module(f'data.dataloaders.{args.dataloader}')
    dataset_process = importlib.import_module(f'data.{args.dataset}.process_data')

    # Create the input data pipeline
    logger.info('Loading the datasets...')
    df_train, df_valid, df_test = dataset_process.data_split(params)

    train_set = data_loader.TrainDataset(df_train, params)
    validation_set = data_loader.TrainDataset(df_valid, params)

    # use GPU if available
    cuda_exist = torch.cuda.is_available()

    # Set random seeds for reproducible experiments if necessary
    if args.fix_seed:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
    if cuda_exist:
        params.device = torch.device('cuda:0')
        logger.info('Using Cuda...')
        torch.backends.cudnn.benchmark = True
        if args.fix_seed:
            torch.cuda.manual_seed_all(0)
        model = net.Net(params).cuda()
    else:
        params.device = torch.device('cpu')
        logger.info('Not using cuda...')
        model = net.Net(params)

    logger.info(f'Model: \n{str(model)}')

    validation_loader = DataLoader(validation_set, batch_size=params.predict_batch,
                                   sampler=RandomSampler(validation_set), num_workers=4)
    test_set = data_loader.TrainDataset(df_test, params)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers=4)
    params.num_val_windows = len(validation_set)
    params.num_test_windows = len(test_set)

    print('Loading complete.')
    print('Starting evaluation...')

    # Reload weights from the saved file
    start_epoch = utils.load_checkpoint(params.model_dir, args.restore_file, model)

    valid_metrics = evaluate(model, validation_loader, params, plot_num=start_epoch - 1, validation=True)
    test_metrics, predict_all = evaluate(model, test_loader, params, plot_num=start_epoch - 1, validation=False)
    save_path = os.path.join(params.model_dir, 'metrics_test_{}.json'.format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
    np.save(os.path.join(params.model_dir, 'predict_all_eval.npy'), predict_all)
