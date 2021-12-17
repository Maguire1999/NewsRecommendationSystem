import torch
import torch.nn as nn
import time
import numpy as np
import os
import datetime
import copy
import importlib
import enlighten

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from news_recommendation.parameters import parse_args
from news_recommendation.dataset import TrainDataset
from news_recommendation.test import evaluate
from news_recommendation.early_stop import EarlyStopping
from news_recommendation.utils import time_since, create_logger, dict2table

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = parse_args()
Model = getattr(importlib.import_module("news_recommendation.model"),
                args.model)

import ipdb


def train():
    writer = SummaryWriter(log_dir=os.path.join(
        args.tensorboard_runs_path,
        f'{args.model}-{args.dataset}',
        f"{datetime.datetime.now().replace(microsecond=0).isoformat()}{'-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}",
    ))

    try:
        pretrained_word_embedding = torch.from_numpy(
            np.load(f'./data/{args.dataset}/pretrained_word_embedding.npy')
        ).float()
    except FileNotFoundError:
        logger.warning('Pretrained word embedding not found')
        pretrained_word_embedding = None

    model = Model(pretrained_word_embedding).to(device)
    logger.info(model)

    start_time = time.time()
    loss_full = []
    step = 0
    early_stopping = EarlyStopping()

    best_checkpoint = copy.deepcopy(model.state_dict())
    best_val_metrics = {}
    if args.save_checkpoint:
        checkpoint_dir = os.path.join(args.checkpoint_path,
                                      f'{args.model}-{args.dataset}')
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    enlighten_manager = enlighten.get_manager()
    batch = 0

    try:
        with enlighten_manager.counter(total=args.num_epochs,
                                       desc='Training epochs',
                                       unit='epochs') as epoch_pbar:
            for epoch in epoch_pbar(range(1, args.num_epochs + 1)):
                dataset = TrainDataset(f'data/{args.dataset}/train.tsv',
                                       f'data/{args.dataset}/news.tsv', epoch,
                                       logger)
                # TODO pin_memory
                dataloader = DataLoader(dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers,
                                        drop_last=True,
                                        pin_memory=True)
                with enlighten_manager.counter(total=len(dataloader),
                                               desc='Training batches',
                                               unit='batches',
                                               leave=False) as batch_pbar:
                    for minibatch in batch_pbar(dataloader):
                        batch += 1

                        single_news_length = list(
                            dataset.news_pattern.values())[-1][-1]
                        history = minibatch[:, dataset.behaviors_pattern[
                            'history'][0]:dataset.behaviors_pattern['history']
                                            [1]].reshape(
                                                -1, single_news_length)

                        positive_candidate = minibatch[:, dataset.
                                                       behaviors_pattern[
                                                           'positive_candidate']
                                                       [0]:dataset.
                                                       behaviors_pattern[
                                                           'positive_candidate']
                                                       [1]]
                        negative_candidates = minibatch[:, dataset.behaviors_pattern[
                            'negative_candidates'][0]:dataset.behaviors_pattern[
                                'negative_candidates'][1]].reshape(
                                    -1, single_news_length)

                        if 'user' in dataset.behaviors_pattern:
                            user = minibatch[:,
                                             dataset.behaviors_pattern['user']
                                             [0]:dataset.
                                             behaviors_pattern['user'][1]]
                        if 'history_length' in dataset.behaviors_pattern:
                            history_length = minibatch[:, dataset.
                                                       behaviors_pattern[
                                                           'history_length']
                                                       [0]:dataset.
                                                       behaviors_pattern[
                                                           'history_length']
                                                       [1]]

                        if args.model == 'LSTUR':
                            loss = model(user, history, history_length,
                                         positive_candidate,
                                         negative_candidates)
                        elif args.model == 'NRMS':
                            loss = model(history, positive_candidate,
                                         negative_candidates)
                        elif args.model == 'NAML':
                            loss = model(history, positive_candidate,
                                         negative_candidates,
                                         dataset.news_pattern)

                        loss_full.append(loss)

                        if batch % 10 == 0:
                            writer.add_scalar('Train/Loss', loss, step)

                        if batch % args.num_batches_show_loss == 0:
                            logger.info(
                                f"Time {time_since(start_time)}, batches {batch}, current loss {loss:.4f}, average loss: {np.mean(loss_full):.4f}, latest average loss: {np.mean(loss_full[-256:]):.4f}"
                            )

                        if batch % args.num_batches_validate == 0:
                            model.eval()
                            metrics = evaluate(model, './data/val',
                                               args.num_workers, 200000)
                            model.train()
                            for metric, value in metrics.items():
                                writer.add_scalar(f'Validation/{metric}',
                                                  value, step)

                            logger.info(
                                f"Time {time_since(start_time)}, batches {batch}, metrics\n{dict2table(metrics)}"
                            )

                            early_stop, get_better = early_stopping(
                                -metrics['AUC'])
                            if early_stop:
                                logger.info('Early stop.')
                                break
                            elif get_better:
                                best_checkpoint = copy.deepcopy(
                                    model.state_dict())
                                best_val_metrics = copy.deepcopy(metrics)
                                if args.save_checkpoint:
                                    torch.save(
                                        model.state_dict(),
                                        os.path.join(checkpoint_dir,
                                                     f"ckpt-{step}.pth"))

    except KeyboardInterrupt:
        logger.info('Stop in advance')

    logger.info(
        f'Best metrics on validation set\n{dict2table(best_val_metrics)}')

    model.load_state_dict(best_checkpoint)
    model.eval()
    metrics = evaluate(model, './data/test', args.num_workers)
    logger.info(f'Metrics on test set\n{dict2table(metrics)}')


if __name__ == '__main__':
    logger = create_logger()
    logger.info(args)
    logger.info(f'Using device: {device}')
    logger.info(f'Training {args.model} on {args.dataset}')
    train()
