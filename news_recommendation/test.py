import numpy as np
import torch
import sys
import os
import importlib

from multiprocessing import Process, SimpleQueue
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from news_recommendation.shared import args, logger, device, enlighten_manager
from news_recommendation.utils import latest_checkpoint, dict2table, calculate_cos_similarity
from news_recommendation.dataset import NewsDataset, UserDataset, EvaluationBehaviorsDataset

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


def calculate_single_user_metric(pair):
    try:
        auc = roc_auc_score(*pair)
        mrr = mrr_score(*pair)
        ndcg5 = ndcg_score(*pair, 5)
        ndcg10 = ndcg_score(*pair, 10)
        return [auc, mrr, ndcg5, ndcg10]
    except ValueError:
        return None


@torch.no_grad()
def evaluate(model, target, max_length=sys.maxsize):
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
    news_vectors = []
    with enlighten_manager.counter(total=len(news_dataloader),
                                   desc='Calculating vectors for news',
                                   leave=False) as pbar:
        for minibatch in pbar(news_dataloader):
            news_vectors.append(
                model.get_news_vector(minibatch.to(device),
                                      news_dataset.news_pattern))
    news_vectors = torch.cat(news_vectors, dim=0)

    if args.show_similarity:
        logger.info(
            f"News cos similarity: {calculate_cos_similarity(news_vectors.cpu().numpy()[1:]):.4f}"
        )

    user_dataset = UserDataset(f'data/{args.dataset}/{target}.tsv')
    user_dataloader = DataLoader(user_dataset,
                                 batch_size=args.batch_size * 16,
                                 shuffle=False,
                                 drop_last=False)

    user_vectors = []
    with enlighten_manager.counter(total=len(user_dataloader),
                                   desc='Calculating vectors for users',
                                   leave=False) as pbar:
        for minibatch in pbar(user_dataloader):
            if args.model == 'LSTUR':
                user_vectors.append(
                    model.get_user_vector(
                        news_vectors[minibatch['history']],
                        news_vectors[minibatch['user']],
                        news_vectors[minibatch['history_length']],
                    ))
            else:
                user_vectors.append(
                    model.get_user_vector(news_vectors[minibatch['history']]))

    user_vectors = torch.cat(user_vectors, dim=0)

    if args.show_similarity:
        logger.info(
            f"User cos similarity: {calculate_cos_similarity(user_vectors.cpu().numpy()):.4f}"
        )

    behaviors_dataset = EvaluationBehaviorsDataset(
        f'data/{args.dataset}/{target}.tsv', user_dataset.user2index)

    if len(behaviors_dataset) > max_length:
        if target == 'test':
            logger.warning(
                'You are slicing the test dataset, the results may not be complete'
            )
        behaviors_dataset = Subset(behaviors_dataset, range(max_length))

    def worker_function(task_queue, result_queue):
        results = []
        for task in iter(task_queue.get, None):
            result = calculate_single_user_metric(task)
            if result is not None:
                results.append(result)

        result_queue.put((len(results), np.average(np.array(results), axis=0)))
        result_queue.put(None)

    task_queue = SimpleQueue()
    result_queue = SimpleQueue()

    workers = []
    for _ in range(args.num_workers):
        worker = Process(target=worker_function,
                         args=(task_queue, result_queue))
        worker.start()
        workers.append(worker)

    with enlighten_manager.counter(
            total=len(behaviors_dataset),
            desc='Calculating metrics with multiprocessing',
            leave=False) as pbar:
        for behaviors in behaviors_dataset:
            pbar.update()

            candidates = behaviors['positive_candidates'] + behaviors[
                'negative_candidates']
            news_vector = news_vectors[candidates]
            user_vector = user_vectors[behaviors['user_index']]
            click_probability = model.get_prediction(news_vector, user_vector)

            y_pred = click_probability.tolist()
            y_true = [1] * len(behaviors['positive_candidates']) + [0] * len(
                behaviors['negative_candidates'])

            task_queue.put((y_true, y_pred))

    for _ in range(args.num_workers):
        task_queue.put(None)

    results = []
    none_count = 0
    while True:
        result = result_queue.get()
        if result is None:
            none_count += 1
            if none_count == args.num_workers:
                assert len(results) == args.num_workers
                break
        else:
            results.append(result)

    for worker in workers:
        worker.join()

    return dict(
        zip(['AUC', 'MRR', 'nDCG@5', 'nDCG@10'],
            np.average(np.array(list(zip(*results))[1]),
                       axis=0,
                       weights=list(zip(*results))[0])))


if __name__ == '__main__':
    logger.info(args)
    logger.info(f'Using device: {device}')
    logger.info(f'Testing {args.model} on {args.dataset}')
    model = Model().to(device)
    checkpoint_path = latest_checkpoint(
        os.path.join(args.checkpoint_dir, f'{args.model}-{args.dataset}'))
    if checkpoint_path is None:
        logger.warning(
            'No checkpoint file found! Evaluating with randomly initiated model'
        )
    else:
        logger.info(f"Load saved parameters in {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
    model.eval()
    metrics = evaluate(model, 'test')
    logger.info(f'Metrics on test set\n{dict2table(metrics)}')
