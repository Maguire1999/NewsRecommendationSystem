from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import BaseDataset
import torch
import torch.nn as nn
import time
import numpy as np
from config import model_name
from tqdm import tqdm
import os
from pathlib import Path
from test import evaluate
import importlib
import datetime
import copy
import math

try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train():
    writer = SummaryWriter(
        log_dir=
        f"./runs/{model_name}/{datetime.datetime.now().replace(microsecond=0).isoformat()}{'-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}"
    )

    try:
        pretrained_word_embedding = torch.from_numpy(
            np.load('./data/train/pretrained_word_embedding.npy')).float()
    except FileNotFoundError:
        pretrained_word_embedding = None

    model = Model(config, pretrained_word_embedding).to(device)

    print(model)

    dataset = BaseDataset('data/train/behaviors_parsed.tsv',
                          'data/train/news_parsed.tsv')

    print(f"Load training dataset with size {len(dataset)}.")

    dataloader = iter(
        DataLoader(dataset,
                   batch_size=config.batch_size,
                   shuffle=True,
                   num_workers=config.num_workers,
                   drop_last=True,
                   pin_memory=True))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    start_time = time.time()
    loss_full = []
    exhaustion_count = 0
    step = 0
    early_stopping = EarlyStopping()

    best_checkpoint = copy.deepcopy(model.state_dict())
    best_val_metrics = {}
    if config.save_checkpoint:
        checkpoint_dir = os.path.join('./checkpoint', model_name)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(
            1,
            config.num_epochs * len(dataset) // config.batch_size + 1),
                  desc="Training"):
        try:
            minibatch = next(dataloader)
        except StopIteration:
            exhaustion_count += 1
            tqdm.write(
                f"Training data exhausted for {exhaustion_count} times after {i} batches, reuse the dataset."
            )
            dataloader = iter(
                DataLoader(dataset,
                           batch_size=config.batch_size,
                           shuffle=True,
                           num_workers=config.num_workers,
                           drop_last=True,
                           pin_memory=True))
            minibatch = next(dataloader)

        step += 1
        if model_name == 'LSTUR':
            y_pred = model(minibatch["user"], minibatch["clicked_news_length"],
                           minibatch["candidate_news"],
                           minibatch["clicked_news"])
        elif model_name == 'TANR':
            y_pred, topic_classification_loss = model(
                minibatch["candidate_news"], minibatch["clicked_news"])
        else:
            y_pred = model(minibatch["candidate_news"],
                           minibatch["clicked_news"])

        y = torch.zeros(len(y_pred)).long().to(device)
        loss = criterion(y_pred, y)

        if model_name == 'TANR':
            if i % 10 == 0:
                writer.add_scalar('Train/BaseLoss', loss.item(), step)
                writer.add_scalar('Train/TopicClassificationLoss',
                                  topic_classification_loss.item(), step)
                writer.add_scalar(
                    'Train/TopicBaseRatio',
                    topic_classification_loss.item() / loss.item(), step)
            loss += config.topic_classification_loss_weight * topic_classification_loss
        loss_full.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            writer.add_scalar('Train/Loss', loss.item(), step)

        if i % config.num_batches_show_loss == 0:
            tqdm.write(
                f"Time {time_since(start_time)}, batches {i}, current loss {loss.item():.4f}, average loss: {np.mean(loss_full):.4f}, latest average loss: {np.mean(loss_full[-256:]):.4f}"
            )

        if i % config.num_batches_validate == 0:
            model.eval()
            metrics = evaluate(model, './data/val', config.num_workers, 200000)
            model.train()
            for metric, value in metrics.items():
                writer.add_scalar(f'Validation/{metric}', value, step)

            tqdm.write(
                f"Time {time_since(start_time)}, batches {i}, metrics\n{dict2table(metrics)}"
            )

            early_stop, get_better = early_stopping(-metrics['AUC'])
            if early_stop:
                tqdm.write('Early stop.')
                break
            elif get_better:
                best_checkpoint = copy.deepcopy(model.state_dict())
                best_val_metrics = copy.deepcopy(metrics)
                if config.save_checkpoint:
                    torch.save(
                        model.state_dict(),
                        os.path.join(checkpoint_dir, f"ckpt-{step}.pth"))

    print(f'Best metrics on validation set\n{dict2table(best_val_metrics)}')

    model.load_state_dict(best_checkpoint)
    model.eval()
    metrics = evaluate(model, './data/test', config.num_workers)
    print(f'Metrics on test set\n{dict2table(metrics)}')


if __name__ == '__main__':
    print('Using device:', device)
    print(f'Training model {model_name}')
    train()
