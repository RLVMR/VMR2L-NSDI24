import argparse
import logging
import os
import importlib
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm

import utils
from evaluate import evaluate
from models.components.Adam import Adam

model_list = utils.model_list()


def train(model: nn.Module,
          optimizer: torch.optim.Optimizer,
          train_loader: DataLoader,
          params: utils.Params) -> float:
    """ Train the model on one epoch by batches.
    Args:
        model: (torch.nn.Module) the neural network model
        optimizer: (torch.optim.Optimizer) optimizer for parameters of model
        train_loader: load train data and labels
        params: (Params) hyperparameters
    Returns:
        loss_epoch: (np.array) loss of each batch in the epoch
    """
    model.train()
    loss_epoch = np.zeros((len(train_loader), params.num_loss))
    # how many steps to use in the loss function, not applicable to some models

    for i, batch in enumerate(tqdm(train_loader)):
        vm_states, pm_states, labels = map(lambda x: x.to(params.device), batch)
        loss_combined, loss = model.do_train(vm_states, pm_states, labels)
        if (i + 1) % 2 == 0:
            optimizer.step()
            optimizer.zero_grad()
        loss_epoch[i] = loss_combined

    loss = loss.item()  # loss per timestep
    logger.info(f'train_loss: {loss}')
    return loss_epoch


def train_and_evaluate(model: nn.Module,
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       test_loader: DataLoader,
                       optimizer: torch.optim.Optimizer,
                       params: utils.Params,
                       restore_file: str = None) -> None:
    """ Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network model
        train_loader: (Dataloader) load train data and labels
        val_loader: (Dataloader) load validation data and labels
        test_loader: (Dataloader) load test data and labels
        optimizer: (torch.optim.Optimizer) optimizer for parameters of model
        params: (Params) hyperparameters
        restore_file: (string) optional - name of file to restore from (without its extension .pth.tar)
    """

    logger.info('Begin training and evaluation...')
    best_epoch = 0
    best_valid_mae = float('inf')
    best_test_mae = float('inf')
    train_len = len(train_loader)
    mae_valid_summary = np.zeros(params.num_epochs)
    mae_test_summary = np.zeros(params.num_epochs)
    loss_summary = np.zeros((train_len * params.num_epochs, params.num_loss))

    # reload weights from restore_file if specified
    if restore_file is not None:
        start_epoch, restored_loss = utils.load_checkpoint(params.model_dir, restore_file, model, optimizer, loss=True)
        loss_summary[:restored_loss.shape[0]] = restored_loss
        logger.info(f'- Restored file (epoch {start_epoch - 1}): evaluation on the validation set')
        valid_metrics = evaluate(model, val_loader, params, plot_num=start_epoch - 1, validation=True)
        logger.info(f'- Restored file (epoch {start_epoch - 1}): evaluation on the test set')
        test_metrics, _ = evaluate(model, test_loader, params, plot_num=start_epoch - 1, validation=False)
        best_valid_mae = valid_metrics['mae']
        best_test_mae = test_metrics['mae']
        logger.info('<<<--- End of evaluation on the restored file --->>>')
        logger.info('<<<--- Continue training --->>>')
    else:
        start_epoch = 0

    for epoch in range(start_epoch, params.num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, params.num_epochs))
        loss_summary[epoch * train_len:(epoch + 1) * train_len] = train(model, optimizer, train_loader, params)

        logger.info('- Evaluation on the validation set')
        valid_metrics = evaluate(model, val_loader, params, plot_num=epoch, validation=True)
        logger.info('- Evaluation on the test set')
        test_metrics, predict_all = evaluate(model, test_loader, params, plot_num=epoch, validation=False)

        mae_valid_summary[epoch] = valid_metrics['mae']
        mae_test_summary[epoch] = test_metrics['mae']
        is_best = valid_metrics['mae'] <= best_valid_mae

        # save weights
        if is_best is True or epoch % 10 == 0:
            """
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict()},
                                  is_best=is_best,
                                  epoch=epoch,
                                  checkpoint=params.model_dir,
                                  loss=loss_summary[:(epoch + 1) * train_len])
            """
            if is_best:
                torch.save({'state_dict': model.state_dict(), 'use_se': True},
                           os.path.join(params.model_dir, 'best.tar'))
            else:
                torch.save({'state_dict': model.state_dict(), 'use_se': True},
                           os.path.join(params.model_dir, f'epoch_{epoch}.tar'))

        if is_best:  # if the current set of weights has best loss on the validation set
            logger.info('- Found new best validation! Updating best test metrics...')
            best_epoch = epoch
            best_valid_mae = valid_metrics['mae']
            best_test_mae = test_metrics['mae']
            best_test_json_path = os.path.join(params.model_dir, 'metrics_test_best_weights.json')
            best_val_json_path = os.path.join(params.model_dir, 'metrics_val_best_weights.json')
            utils.save_dict_to_json(test_metrics, best_test_json_path)
            utils.save_dict_to_json(valid_metrics, best_val_json_path)
            np.save(os.path.join(params.model_dir, 'predict_all_best.npy'), predict_all)

        logger.info(f'Current Best MAE is: {best_test_mae:05.5f}, from epoch {best_epoch}')
        logger.info(f'Current Best validation p50 is {best_valid_mae:05.5f}')

        utils.plot_all_epoch(mae_test_summary[:epoch + 1], mae_valid_summary[:epoch + 1],
                             params.dataset + '_mae', params.plot_title, params.plot_dir, plot_start=params.plot_start)
        utils.plot_all_loss(loss_summary[:(epoch + 1) * train_len], params.dataset + '_loss',
                            params.plot_title, params.plot_dir)

        last_test_json_path = os.path.join(params.model_dir, 'metrics_test_last_weights.json')
        last_val_json_path = os.path.join(params.model_dir, 'metrics_val_last_weights.json')
        utils.save_dict_to_json(test_metrics, last_test_json_path)
        utils.save_dict_to_json(valid_metrics, last_val_json_path)
        np.save(os.path.join(params.model_dir, 'mae_test_summary'), mae_test_summary)
        np.save(os.path.join(params.model_dir, 'loss_summary'), loss_summary)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='vm_shuffled_extra_graph', help='Name of the dataset')
    parser.add_argument('--dataloader', default='enc_dec_attn', help='Which data loader to use')
    parser.add_argument('--model', default='baseline_trans', choices=model_list, help='Which model to use')
    parser.add_argument('--model-dir', default='base_model',
                        help='Directory containing params.json, and training results')
    parser.add_argument('--param-set', default=None, help='Set of model parameters created for hypersearch')
    parser.add_argument('--fix-seed', action='store_true', help='Whether to fix random seed')
    parser.add_argument('--save-best', action='store_true', help='Whether to save best MAE to param_search.txt')
    parser.add_argument('--restore-file', default=None,
                        help='Optional, name of the file in --model_dir containing weights to reload before \
                        training')  # 'best' or 'epoch_#'

    args = parser.parse_args()

    # Load the parameters from json file
    data_dir = os.path.join('data', args.dataset)
    data_json_path = os.path.join(data_dir, 'params.json')
    assert os.path.isfile(data_json_path), f'No data json config file found at {data_json_path}'
    params = utils.Params(data_json_path)

    if args.param_set is not None:
        params.model_dir = os.path.join('experiments', args.model_dir, args.dataset, args.model,
                                        args.dataloader, args.param_set)
    else:
        params.model_dir = os.path.join('experiments', args.model_dir, args.dataset, args.model, args.dataloader)
    json_path = os.path.join(params.model_dir, 'params.json')
    assert os.path.isfile(json_path), f'No model json configuration file found at {json_path}'
    params.update(json_path)

    utils.set_logger(os.path.join(params.model_dir, 'train.log'))
    logger = logging.getLogger(f'VM.{args.model}')

    params.dataset = args.dataset
    params.model = args.model
    params.plot_dir = os.path.join(params.model_dir, 'figures')
    if args.param_set is not None:
        params.plot_title = os.path.join(args.model_dir, args.dataset, args.model, args.dataloader,
                                         args.param_set)
    else:
        params.plot_title = os.path.join(args.model_dir, args.dataset, args.model, args.dataloader)

    net = importlib.import_module(f'models.{args.model}')
    data_loader = importlib.import_module(f'data.dataloaders.{args.dataloader}')
    dataset_process = importlib.import_module(f'data.{args.dataset}.process_data')

    # create missing directories
    try:
        os.mkdir(params.plot_dir)
    except FileExistsError:
        pass

    print('Building the datasets...')
    df_train, df_valid, df_test = dataset_process.data_split(params)

    train_set = data_loader.TrainDataset(df_train, params)
    validation_set = data_loader.TrainDataset(df_valid, params)

    if params.weighted_sampler:
        logger.info('Using weighted sampler.')
        # Use weighted sampler instead of random sampler
        sampler = data_loader.WeightedSampler(train_set.target)
        train_loader = DataLoader(train_set, batch_size=params.batch_size, shuffle=False, sampler=sampler,
                                  num_workers=4, drop_last=True)
    else:
        logger.info('Using random sampler.')
        train_loader = DataLoader(train_set, batch_size=params.batch_size, shuffle=False,
                                  sampler=RandomSampler(train_set), num_workers=4, drop_last=True)

    validation_loader = DataLoader(validation_set, batch_size=params.predict_batch,
                                   sampler=SequentialSampler(validation_set), num_workers=4)

    test_set = data_loader.TrainDataset(df_test, params)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=SequentialSampler(test_set),
                             num_workers=4)
    params.num_val_windows = len(validation_set)
    params.num_test_windows = len(test_set)

    print('Loading complete.')

    # use GPU if available
    cuda_exist = torch.cuda.is_available()
    # cuda_exist = False

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

    # Prepare optimizer and schedule (linear warmup and decay), from huggingface/transformers
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    num_training_steps = int(len(train_set) // params.batch_size * params.num_epochs)
    logger.info('=== Using OpenAI Adam as the optimizer ===')  # Adapted from OpenAI
    optimizer = Adam(optimizer_grouped_parameters,
                     lr=params.learning_rate,
                     warmup=params.warmup_portion,
                     t_total=num_training_steps,
                     max_grad_norm=params.max_grad_norm)

    # Train the model
    logger.info('Starting training for {} epoch(s)'.format(params.num_epochs))
    train_and_evaluate(model,
                       train_loader,
                       validation_loader,
                       test_loader,
                       optimizer,
                       params,
                       args.restore_file)
