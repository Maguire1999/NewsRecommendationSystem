import torch
import time
import numpy as np
import os
import datetime
import copy
import importlib
import random

from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from news_recommendation.shared import args
from news_recommendation.dataset import TrainDataset
from news_recommendation.test import evaluate
from news_recommendation.utils import time_since, dict2table, EarlyStopping
from news_recommendation.shared import args, logger, device, enlighten_manager
from news_recommendation.model.general.trainer.centralized import CentralizedModelTrainer
from news_recommendation.model.general.trainer.federated import FederatedModelTrainer

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

    start_time = time.time()
    loss_full = []
    early_stopping = EarlyStopping(patience=args.patience)
    best_checkpoint = copy.deepcopy(model.state_dict())
    best_val_metrics = {}
    if args.save_checkpoint:
        checkpoint_dir = os.path.join(args.checkpoint_dir,
                                      f'{args.model}-{args.dataset}')
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    unit = 0

    datasets = {}
    try:
        with enlighten_manager.counter(total=args.num_epochs,
                                       desc='Training epochs',
                                       unit='epochs',
                                       leave=False) as epoch_pbar:
            for epoch in epoch_pbar(range(args.num_epochs)):
                epoch_hash = epoch % args.max_epoch_cache_num
                if epoch_hash in datasets:
                    dataset = datasets[epoch_hash]
                else:
                    dataset = TrainDataset(f'data/{args.dataset}/train.tsv',
                                           f'data/{args.dataset}/news.tsv',
                                           epoch_hash)
                    datasets[epoch_hash] = dataset

                if isinstance(model, CentralizedModelTrainer):
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
                            unit += 1

                            loss = model(minibatch, dataset.news_pattern)
                            loss_full.append(loss)

                            if unit % args.num_units_record_loss == 0:
                                writer.add_scalar('Train/Loss', loss, unit)

                            if unit % args.num_units_show_loss == 0:
                                logger.info(
                                    f"Time {time_since(start_time)}, epoch {epoch}, batch {unit}, current loss {loss:.4f}, average loss {np.mean(loss_full):.4f}, latest average loss {np.mean(loss_full[-10:]):.4f}"
                                )

                    if (epoch + 1) % args.num_epochs_validate == 0:
                        model.eval()
                        metrics = evaluate(model, 'val', 200000)
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

                elif isinstance(model, FederatedModelTrainer):
                    if epoch == 0:
                        # Only need do once
                        user2indexs = {}
                        for i, user in enumerate(
                                dataset.data['user'].tolist()):
                            if user not in user2indexs:
                                user2indexs[user] = [i]
                            else:
                                user2indexs[user].append(i)

                    users = random.sample(user2indexs.keys(),
                                          args.num_users_per_epoch)

                    old_model = copy.deepcopy(model.state_dict())
                    new_model = {}
                    with enlighten_manager.counter(total=len(users),
                                                   desc='Training users',
                                                   unit='users',
                                                   leave=False) as user_pbar:
                        for user in user_pbar(users):
                            unit += 1

                            minibatch = dataset[user2indexs[user]]
                            model.load_state_dict(old_model)
                            loss = model(minibatch, dataset.news_pattern)

                            for k, v in model.state_dict().items():
                                if k not in new_model:
                                    new_model[k] = v
                                else:
                                    new_model[k] += v

                            loss_full.append(loss)

                            if unit % args.num_units_record_loss == 0:
                                writer.add_scalar('Train/Loss', loss, unit)

                            if unit % args.num_units_show_loss == 0:
                                logger.info(
                                    f"Time {time_since(start_time)}, epoch {epoch}, user {unit}, current loss {loss:.4f}, average loss {np.mean(loss_full):.4f}, latest average loss {np.mean(loss_full[-10:]):.4f}"
                                )

                    for k in new_model.keys():
                        new_model[k] = new_model[k] / len(users)

                    model.load_state_dict(new_model)

                    if (epoch + 1) % args.num_epochs_validate == 0:
                        model.eval()
                        metrics = evaluate(model, 'val', 200000)
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

                else:
                    raise NotImplementedError

    except KeyboardInterrupt:
        logger.info('Stop in advance')

    logger.info(
        f'Best metrics on validation set\n{dict2table(best_val_metrics)}')

    model.load_state_dict(best_checkpoint)
    model.eval()
    metrics = evaluate(model, 'test')
    logger.info(f'Metrics on test set\n{dict2table(metrics)}')


if __name__ == '__main__':
    logger.info(args)
    logger.info(f'Using device: {device}')
    logger.info(f'Training {args.model} on {args.dataset}')
    train()
