import numpy as np
import torch
import sys
import pandas as pd

from tqdm import tqdm
from os import path
from ast import literal_eval
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

from news_recommendation.shared import args, logger, device


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def value2rank(d):
    values = list(d.values())
    ranks = [sorted(values, reverse=True).index(x) for x in values]
    return {k: ranks[i] + 1 for i, k in enumerate(d.keys())}


def calculate_single_user_metric(pair):
    try:
        auc = roc_auc_score(*pair)
        mrr = mrr_score(*pair)
        ndcg5 = ndcg_score(*pair, 5)
        ndcg10 = ndcg_score(*pair, 10)
        return [auc, mrr, ndcg5, ndcg10]
    except ValueError:
        return [np.nan] * 4


@torch.no_grad()
def evaluate(model, directory, num_workers, max_count=sys.maxsize):
    """
    Evaluate model on target directory.
    Args:
        model: model to be evaluated
        directory: the directory that contains two files (behaviors.tsv, news_parsed.tsv)
        num_workers: processes number for calculating metrics
    Returns:
        AUC
        MRR
        nDCG@5
        nDCG@10
    """
    news_dataset = NewsDataset(path.join(directory, 'news_parsed.tsv'))
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=args.batch_size * 16,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 drop_last=False,
                                 pin_memory=True)

    news2vector = {}
    for minibatch in tqdm(news_dataloader,
                          desc="Calculating vectors for news"):
        news_ids = minibatch["id"]
        if any(id not in news2vector for id in news_ids):
            news_vector = model.get_news_vector(minibatch)
            for id, vector in zip(news_ids, news_vector):
                if id not in news2vector:
                    news2vector[id] = vector

    news2vector['PADDED_NEWS'] = torch.zeros(
        list(news2vector.values())[0].size())

    user_dataset = UserDataset(path.join(directory, 'behaviors.tsv'),
                               'data/train/user2int.tsv')
    user_dataloader = DataLoader(user_dataset,
                                 batch_size=args.batch_size * 16,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 drop_last=False,
                                 pin_memory=True)

    user2vector = {}
    for minibatch in tqdm(user_dataloader,
                          desc="Calculating vectors for users"):
        user_strings = minibatch["clicked_news_string"]
        if any(user_string not in user2vector for user_string in user_strings):
            history_vector = torch.stack([
                torch.stack([news2vector[x].to(device) for x in news_list],
                            dim=0) for news_list in minibatch["clicked_news"]
            ],
                                         dim=0).transpose(0, 1)
            if model == 'LSTUR':
                user_vector = model.get_user_vector(
                    minibatch['user'], minibatch['clicked_news_length'],
                    history_vector)
            else:
                user_vector = model.get_user_vector(history_vector)
            for user, vector in zip(user_strings, user_vector):
                if user not in user2vector:
                    user2vector[user] = vector

    behaviors_dataset = BehaviorsDataset(path.join(directory, 'behaviors.tsv'))
    behaviors_dataloader = DataLoader(behaviors_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=args.num_workers)

    count = 0

    tasks = []

    for minibatch in tqdm(behaviors_dataloader,
                          desc="Calculating probabilities"):
        count += 1
        if count == max_count:
            break

        candidate_news_vector = torch.stack([
            news2vector[news[0].split('-')[0]]
            for news in minibatch['impressions']
        ],
                                            dim=0)
        user_vector = user2vector[minibatch['clicked_news_string'][0]]
        click_probability = model.get_prediction(candidate_news_vector,
                                                 user_vector)

        y_pred = click_probability.tolist()
        y_true = [
            int(news[0].split('-')[1]) for news in minibatch['impressions']
        ]

        tasks.append((y_true, y_pred))

    with Pool(processes=num_workers) as pool:
        results = pool.map(calculate_single_user_metric, tasks)

    aucs, mrrs, ndcg5s, ndcg10s = np.array(results).T

    return {
        'AUC': np.nanmean(aucs),
        'MRR': np.nanmean(mrrs),
        'nDCG@5': np.nanmean(ndcg5s),
        'nDCG@10': np.nanmean(ndcg10s)
    }
    # TODO:
    # return dict(
    #     zip(['AUC', 'MRR', 'nDCG@5', 'nDCG@10'],
    #         np.nanmean(np.array(results), axis=1)))


if __name__ == '__main__':
    logger.info('Using device:', device)
    logger.info(f'Evaluating model {model}')
    # Don't need to load pretrained word embedding
    # since it will be loaded from checkpoint later
    model = Model(config).to(device)
    from train import latest_checkpoint  # Avoid circular imports
    checkpoint_path = latest_checkpoint(path.join('./checkpoint', model))
    if checkpoint_path is None:
        logger.error('No checkpoint file found!')
        exit()
    logger.info(f"Load saved parameters in {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()
    metrics = evaluate(model, './data/test', args.num_workers)
    logger.info(f'Metrics on test set\n{dict2table(metrics)}')
