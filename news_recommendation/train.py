import torch
import time
import numpy as np
import os
import datetime
import copy
import importlib
import random

from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler, Subset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from news_recommendation.dataset import TrainingDataset
from news_recommendation.test import evaluate
from news_recommendation.utils import time_since, dict2table, EarlyStopping
from news_recommendation.shared import args, logger, device, enlighten_manager
from news_recommendation.model.general.trainer.centralized import CentralizedModel
from news_recommendation.model.general.trainer.federated import FederatedModel

Model = getattr(
    importlib.import_module(f"news_recommendation.model.{args.model}"),
    args.model)


def train():
    writer = SummaryWriter(log_dir=os.path.join(
        args.tensorboard_runs_dir,
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

    if isinstance(model, CentralizedModel):
        model.init_backprop(model.parameters())

    start_time = time.time()
    loss_full = []
    early_stopping = EarlyStopping(patience=args.patience)
    best_checkpoint = None
    best_val_metrics = None
    if args.save_checkpoint:
        checkpoint_dir = os.path.join(args.checkpoint_dir,
                                      f'{args.model}-{args.dataset}')
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    datasets = {}
    try:
        if isinstance(model, CentralizedModel):
            batch = 0
            with enlighten_manager.counter(total=args.num_epochs,
                                           desc='Training epochs',
                                           unit='epochs',
                                           leave=False) as epoch_pbar:
                for epoch in epoch_pbar(range(args.num_epochs)):
                    epoch_hash = epoch % args.max_training_dataset_cache_num
                    if epoch_hash in datasets:
                        dataset = datasets[epoch_hash]
                    else:
                        dataset = TrainingDataset(
                            f'data/{args.dataset}/train.tsv',
                            f'data/{args.dataset}/news.tsv', epoch_hash)
                        datasets[epoch_hash] = dataset

                    # Use `sampler=BatchSampler(...)` to support batch indexing of dataset, which is faster
                    dataloader = DataLoader(
                        dataset,
                        sampler=BatchSampler(
                            RandomSampler(dataset)
                            if args.shuffle else SequentialSampler(dataset),
                            batch_size=args.batch_size,
                            drop_last=False,
                        ),
                        collate_fn=lambda x: x[0],
                        pin_memory=True,
                    )

                    with enlighten_manager.counter(total=len(dataloader),
                                                   desc='Training batches',
                                                   unit='batches',
                                                   leave=False) as batch_pbar:
                        for minibatch in batch_pbar(dataloader):
                            if args.dry_run:
                                continue

                            minibatch = {
                                k: v.to(device)
                                for k, v in minibatch.items()
                            }
                            y_pred = model(minibatch, dataset.news_pattern)
                            loss = model.backward(y_pred)
                            loss_full.append(loss)

                            if batch % args.num_batches_show_loss == 0:
                                writer.add_scalar('Train/Loss', loss, batch)
                                logger.info(
                                    f"Time {time_since(start_time)}, epoch {epoch}, batch {batch}, current loss {loss:.4f}, average loss {np.mean(loss_full):.4f}, latest average loss {np.mean(loss_full[-10:]):.4f}"
                                )
                            batch += 1

                    if epoch % args.num_epochs_validate == 0:
                        model.eval()
                        metrics = evaluate(model, 'val')
                        model.train()
                        for metric, value in metrics.items():
                            writer.add_scalar(f'Validation/{metric}', value,
                                              epoch)

                        logger.info(
                            f"Time {time_since(start_time)}, epoch {epoch}, metrics\n{dict2table(metrics)}"
                        )

                        early_stop, get_better = early_stopping(
                            -metrics['AUC'])
                        if early_stop:
                            logger.info('Early stop.')
                            break
                        elif get_better:
                            best_checkpoint = copy.deepcopy(model.state_dict())
                            best_val_metrics = copy.deepcopy(metrics)
                            if args.save_checkpoint:
                                torch.save(
                                    model.state_dict(),
                                    os.path.join(checkpoint_dir,
                                                 f"ckpt-{epoch}.pt"))

        elif isinstance(model, FederatedModel):
            with enlighten_manager.counter(total=args.num_rounds,
                                           desc='Training rounds',
                                           unit='rounds',
                                           leave=False) as round_pbar:
                for round in round_pbar(range(args.num_rounds)):
                    round_hash = round % args.max_training_dataset_cache_num
                    if round_hash in datasets:
                        dataset, user2indexs = datasets[round_hash]
                    else:
                        dataset = TrainingDataset(
                            f'data/{args.dataset}/train.tsv',
                            f'data/{args.dataset}/news.tsv', round_hash)
                        user2indexs = {}
                        for i, user in enumerate(
                                dataset.data['user'].tolist()):
                            if user not in user2indexs:
                                user2indexs[user] = [i]
                            else:
                                user2indexs[user].append(i)
                        datasets[round_hash] = dataset, user2indexs

                    users = random.sample(user2indexs.keys(),
                                          args.num_users_per_round)

                    old_model = copy.deepcopy(model.state_dict())
                    new_model = {}

                    loss = 0
                    with enlighten_manager.counter(total=len(users),
                                                   desc='Training users',
                                                   unit='users',
                                                   leave=False) as user_pbar:
                        for user in user_pbar(users):
                            if args.dry_run:
                                continue

                            model.load_state_dict(old_model)
                            model.init_backprop(model.parameters())

                            dataloader = DataLoader(
                                Subset(dataset, user2indexs[user]),
                                batch_size=args.batch_size,
                                drop_last=False,
                                shuffle=args.shuffle,
                            )

                            for minibatch in dataloader:
                                y_pred = model(minibatch, dataset.news_pattern)
                                loss += model.backward(y_pred) * y_pred.size(0)

                            for k, v in model.state_dict().items():
                                if k not in new_model:
                                    new_model[k] = v * len(user2indexs[user])
                                else:
                                    new_model[k] += v * len(user2indexs[user])

                    for k in new_model.keys():
                        new_model[k] /= sum(
                            len(user2indexs[user]) for user in users)
                    model.load_state_dict(new_model)

                    loss /= sum(len(user2indexs[user]) for user in users)
                    loss_full.append(loss)

                    if round % args.num_rounds_show_loss == 0:
                        writer.add_scalar('Train/Loss', loss, round)
                        logger.info(
                            f"Time {time_since(start_time)}, round {round}, current loss {loss:.4f}, average loss {np.mean(loss_full):.4f}, latest average loss {np.mean(loss_full[-10:]):.4f}"
                        )

                    if round != 0 and round % args.num_rounds_validate == 0:
                        model.eval()
                        metrics = evaluate(model, 'val')
                        model.train()
                        for metric, value in metrics.items():
                            writer.add_scalar(f'Validation/{metric}', value,
                                              round)

                        logger.info(
                            f"Time {time_since(start_time)}, round {round}, metrics\n{dict2table(metrics)}"
                        )

                        early_stop, get_better = early_stopping(
                            -metrics['AUC'])
                        if early_stop:
                            logger.info('Early stop.')
                            break
                        elif get_better:
                            best_checkpoint = copy.deepcopy(model.state_dict())
                            best_val_metrics = copy.deepcopy(metrics)
                            if args.save_checkpoint:
                                torch.save(
                                    model.state_dict(),
                                    os.path.join(checkpoint_dir,
                                                 f"ckpt-{round}.pt"))

        else:
            raise NotImplementedError

    except KeyboardInterrupt:
        logger.info('Stop in advance')

    if best_val_metrics is not None:
        logger.info(
            f'Best metrics on validation set\n{dict2table(best_val_metrics)}')
    if best_checkpoint is not None:
        model.load_state_dict(best_checkpoint)
    model.eval()
    metrics = evaluate(model, 'test')
    logger.info(f'Metrics on test set\n{dict2table(metrics)}')


if __name__ == '__main__':
    logger.info(args)
    logger.info(f'Using device: {device}')
    logger.info(f'Training {args.model} on {args.dataset}')
    train()
