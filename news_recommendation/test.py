import numpy as np
import torch
import os
import importlib

from torch.multiprocessing import Process, SimpleQueue, set_start_method
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

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
        return [np.nan] * 4


def scoring_worker_fn(index, task_queue, mode, news_vectors, user_vectors,
                      prediction_fn):
    behaviors_dataset = EvaluationBehaviorsDataset(
        f'data/{args.dataset}/{mode}.tsv', {}, index, args.num_scoring_workers)

    for behaviors in behaviors_dataset:
        candidates = behaviors['positive_candidates'] + behaviors[
            'negative_candidates']
        news_vector = news_vectors[candidates]
        user_vector = user_vectors[behaviors['user_index']]
        click_probability = prediction_fn(news_vector, user_vector)

        y_pred = click_probability.tolist()
        y_true = [1] * len(behaviors['positive_candidates']) + [0] * len(
            behaviors['negative_candidates'])

        task_queue.put((y_true, y_pred))


def metrics_worker_fn(task_queue, result_queue):
    for task in iter(task_queue.get, None):
        result_queue.put(calculate_single_user_metric(task))


@torch.no_grad()
def evaluate(model, mode):
    """
    Args:

    Returns:
        AUC
        MRR
        nDCG@5
        nDCG@10
    """
    assert mode in ['val', 'test']
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

    user_dataset = UserDataset(f'data/{args.dataset}/{mode}.tsv')
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

    behaviors_count = 0
    for i in range(args.num_scoring_workers):
        # Make sure the cache exists, so in `scoring_worker_fn`, the `user2index` parameter
        # for `EvaluationBehaviorsDataset` can be empty.
        # In this way, `user_dataset.user2index` does not to be passed to `scoring_worker_fn`,
        # saving a lot time on pickling/unpickling
        behaviors_count += len(
            EvaluationBehaviorsDataset(f'data/{args.dataset}/{mode}.tsv',
                                       user_dataset.user2index, i,
                                       args.num_scoring_workers))
    """
    Evaluation with multiprocessing:

                                                  ┌──────────────────┐
                                                  │ Metrics Worker 0 │
                                                  └──────────────────┘

                                                  ┌──────────────────┐
                                                  │ Metrics Worker 1 │
    ┌──────────────────┐                          └──────────────────┘
    │ Scoring Worker 0 │
    └──────────────────┘                          ┌──────────────────┐
                                                  │ Metrics Worker 2 │
    ┌──────────────────┐                          └──────────────────┘
    │ Scoring Worker 1 │       TASK QUEUE                                     RESULT QUEUE
    └──────────────────┘   ───────────────────►   ┌──────────────────┐     ───────────────────►
                                                  │ Metrics Worker 3 │
    ┌──────────────────┐                          └──────────────────┘
    │ Scoring Worker 2 │
    └──────────────────┘                          ┌──────────────────┐
                                                  │ Metrics Worker 4 │
                                                  └──────────────────┘

                                                  ┌──────────────────┐
                                                  │ Metrics Worker 5 │
                                                  └──────────────────┘
    """
    task_queue = SimpleQueue()
    result_queue = SimpleQueue()

    scoring_workers = []

    for i in range(args.num_scoring_workers):
        worker = Process(target=scoring_worker_fn,
                         args=(
                             i,
                             task_queue,
                             mode,
                             news_vectors,
                             user_vectors,
                             model.get_prediction,
                         ))
        worker.start()
        scoring_workers.append(worker)

    metrics_workers = []
    for _ in range(args.num_metrics_workers):
        worker = Process(target=metrics_worker_fn,
                         args=(task_queue, result_queue))
        worker.start()
        metrics_workers.append(worker)

    # wait for the first result to get a more accurate progress bar :)
    results = [result_queue.get()]
    with enlighten_manager.counter(
            count=len(results),
            total=behaviors_count,
            desc='Calculating metrics with multiprocessing',
            leave=False) as pbar:
        while True:
            results.append(result_queue.get())
            pbar.update()
            if len(results) == behaviors_count:
                break

    for worker in scoring_workers:
        worker.join()
    for _ in range(args.num_metrics_workers):
        task_queue.put(None)
    for worker in metrics_workers:
        worker.join()

    return dict(
        zip(['AUC', 'MRR', 'nDCG@5', 'nDCG@10'],
            np.nanmean(np.array(results), axis=0)))


if __name__ == '__main__':
    set_start_method('spawn')
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
