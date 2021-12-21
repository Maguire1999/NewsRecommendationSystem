import numpy as np
import torch
import sys
import os
import importlib

from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from news_recommendation.shared import args, logger, device, enlighten_manager
from news_recommendation.utils import latest_checkpoint, dict2table
from news_recommendation.dataset import NewsDataset, UserDataset, BehaviorsDataset

Model = getattr(
    importlib.import_module(f"news_recommendation.model.{args.model}"),
    args.model)


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
def evaluate(model, target, max_count=sys.maxsize):
    """
    Args:
        model: model to be evaluated
        directory: the directory that contains two files (behaviors.tsv, news_parsed.tsv)
    Returns:
        AUC
        MRR
        nDCG@5
        nDCG@10
    """
    assert target in ['val', 'test']
    news_dataset = NewsDataset(f'data/{args.dataset}/news.tsv')
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=args.batch_size * 16,
                                 shuffle=False,
                                 drop_last=False)
    news_vector = []
    with enlighten_manager.counter(total=len(news_dataloader),
                                   desc='Calculating vectors for news',
                                   leave=False) as pbar:
        for minibatch in pbar(news_dataloader):
            news_vector.append(model.get_news_vector(minibatch))
    news_vector = torch.cat(news_vector, dim=0)

    user_dataset = UserDataset(f'data/{args.dataset}/{target}.tsv')
    user_dataloader = DataLoader(user_dataset,
                                 batch_size=args.batch_size * 16,
                                 shuffle=False,
                                 drop_last=False)

    user2vector = {}
    with enlighten_manager.counter(total=len(user_dataloader),
                                   desc='Calculating vectors for users',
                                   leave=False) as pbar:
        for minibatch in pbar(user_dataloader):
            if args.model == 'LSTUR':
                user_vector = model.get_user_vector(
                    news_vector[minibatch['history']],
                    news_vector[minibatch['user']],
                    news_vector[minibatch['history_length']],
                )
            else:
                user_vector = model.get_user_vector(
                    news_vector[minibatch['history']])
            for key, vector in zip(minibatch['key'], user_vector):
                user2vector[key] = vector

    behaviors_dataset = BehaviorsDataset(f'data/{args.dataset}/{target}.tsv')

    count = 0

    tasks = []

    with enlighten_manager.counter(
            total=len(behaviors_dataset),
            desc='Adding tasks for calculating probabilities',
            leave=False) as pbar:
        for behaviors in behaviors_dataset:
            pbar.update()
            count += 1
            if count == max_count:
                break

            candidates = behaviors['positive_candidates'] + behaviors[
                'negative_candidates']
            candidates_vector = news_vector[candidates]
            user_vector = user2vector[behaviors['key']]
            click_probability = model.get_prediction(candidates_vector,
                                                     user_vector)

            y_pred = click_probability.tolist()
            y_true = [1] * len(behaviors['positive_candidates']) + [0] * len(
                behaviors['negative_candidates'])

            tasks.append((y_true, y_pred))

    logger.info('Calculating probabilities with multiprocessing')
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(calculate_single_user_metric, tasks)

    return dict(
        zip(['AUC', 'MRR', 'nDCG@5', 'nDCG@10'],
            np.nanmean(np.array(results), axis=0)))


if __name__ == '__main__':
    logger.info(args)
    logger.info(f'Using device: {device}')
    logger.info(f'Testing {args.model} on {args.dataset}')
    model = Model().to(device)
    checkpoint_path = latest_checkpoint(
        os.path.join(args.checkpoint_dir, f'{args.model}-{args.dataset}'))
    if checkpoint_path is None:
        logger.error('No checkpoint file found!')
        exit()
    logger.info(f"Load saved parameters in {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()
    metrics = evaluate(model, 'test')
    logger.info(f'Metrics on test set\n{dict2table(metrics)}')
