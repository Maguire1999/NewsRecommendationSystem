import pandas as pd
import numpy as np
import torch
import random

from ast import literal_eval
from torch.utils.data import Dataset
from news_recommendation.shared import args, logger
from news_recommendation.utils import load_from_cache


class TrainDataset(Dataset):
    def __init__(self, behaviors_path, news_path, epoch=1):
        super().__init__()

        self.news, self.news_pattern = load_from_cache(
            [
                args.dataset,
                args.num_words_title,
                args.num_words_abstract,
                args.word_frequency_threshold,
                args.dataset_attributes,
                news_path,
            ],
            lambda: self._process_news(news_path),
            args.cache_dir,
            args.cache_dataset,
            lambda x: logger.info(f'Load news cache from {x}'),
            lambda x: logger.info(f'Save news cache to {x}'),
        )
        self.behaviors, self.behaviors_pattern = load_from_cache(
            [
                args.dataset,
                args.num_history,
                args.num_words_title,
                args.num_words_abstract,
                args.word_frequency_threshold,
                args.negative_sampling_ratio,
                args.dataset_attributes,
                behaviors_path,
                epoch,
            ],
            lambda: self._process_behaviors(behaviors_path),
            args.cache_dir,
            args.cache_dataset,
            lambda x: logger.info(
                f'Load training behaviors (epoch {epoch}) cache from {x}'),
            lambda x: logger.info(
                f'Save training behaviors (epoch {epoch}) cache to {x}'),
        )

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, i):
        return self.behaviors[i]

    @staticmethod
    def _process_news(news_path):
        news = pd.read_table(
            news_path,
            usecols=args.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(args.dataset_attributes['news'])
                & set(['title', 'abstract'])
            })
        news_attributes2length = {
            'category': 1,
            'subcategory': 1,
            'title': args.num_words_title,
            'abstract': args.num_words_abstract
        }
        news_elements = []
        news_pattern = {}
        current_length = 0

        for attribute in args.dataset_attributes['news']:
            if attribute in ['category', 'subcategory']:
                news_elements.append(news[attribute].to_numpy()[...,
                                                                np.newaxis])
            elif attribute in ['title', 'abstract']:
                news_elements.append(np.array(news[attribute].tolist()))
            else:
                raise ValueError
            news_pattern[attribute] = (current_length, current_length +
                                       news_attributes2length[attribute])
            current_length += news_attributes2length[attribute]

        news = np.concatenate(news_elements, axis=1)
        # Add a news with id as 0 and content as 0s
        news = np.insert(news, 0, 0, axis=0)
        news = torch.from_numpy(news)
        return news, news_pattern

    def _process_behaviors(self, behaviors_path):
        behaviors = pd.read_table(
            behaviors_path,
            converters={
                attribute: literal_eval
                for attribute in
                ['history', 'positive_candidates', 'negative_candidates']
            })
        behaviors = behaviors.explode('positive_candidates', ignore_index=True)
        behaviors.positive_candidates = behaviors.positive_candidates.infer_objects(
        )

        def sample_negatives(negatives):
            if len(negatives) > args.negative_sampling_ratio:
                return random.sample(negatives, args.negative_sampling_ratio)
            else:
                return negatives + [0] * (args.negative_sampling_ratio -
                                          len(negatives))

        behaviors.negative_candidates = behaviors.negative_candidates.apply(
            sample_negatives)

        single_news_length = list(self.news_pattern.values())[-1][-1]
        behaviors_attributes2length = {
            'user':
            1,
            'history':
            args.num_history * single_news_length,
            'history_length':
            1,
            'positive_candidates':
            single_news_length,
            'negative_candidates':
            args.negative_sampling_ratio * single_news_length,
        }
        behaviors_elements = []
        behaviors_pattern = {}
        current_length = 0

        for attribute in args.dataset_attributes['behaviors']:
            if attribute in ['user', 'history_length', 'positive_candidates']:
                numpy_array = behaviors[attribute].to_numpy()[..., np.newaxis]
            elif attribute in ['history', 'negative_candidates']:
                numpy_array = np.array(behaviors[attribute].tolist())
            else:
                raise ValueError

            if attribute in [
                    'history', 'positive_candidates', 'negative_candidates'
            ]:
                numpy_array = self.news[numpy_array]
                numpy_array = numpy_array.reshape((numpy_array.shape[0], -1))

            behaviors_elements.append(numpy_array)
            behaviors_pattern[attribute] = (
                current_length,
                current_length + behaviors_attributes2length[attribute])
            current_length += behaviors_attributes2length[attribute]

        behaviors = np.concatenate(behaviors_elements, axis=1)
        behaviors = torch.from_numpy(
            behaviors)  # TODO what if loaded from pickle and use gpu
        return behaviors, behaviors_pattern


class NewsDataset(Dataset):
    """
    Load news for evaluation.
    """
    def __init__(self, news_path):
        super().__init__()
        self.news, self.news_pattern = load_from_cache(
            [
                args.dataset,
                args.num_words_title,
                args.num_words_abstract,
                args.word_frequency_threshold,
                args.dataset_attributes,
                news_path,
            ],
            lambda: TrainDataset._process_news(news_path),
            args.cache_dir,
            args.cache_dataset,
            lambda x: logger.info(f'Load news cache from {x}'),
            lambda x: logger.info(f'Save news cache to {x}'),
        )

    def __len__(self):
        return len(self.news)

    def __getitem__(self, i):
        return self.news[i]


class UserDataset(Dataset):
    """
    Load users for evaluation, duplicated rows will be dropped
    """
    def __init__(self, behaviors_path):
        super().__init__()
        behaviors = pd.read_table(
            behaviors_path,
            usecols=set(args.dataset_attributes['behaviors'])
            & set(['user', 'history', 'history_length']),
        ).drop_duplicates(ignore_index=True)
        self.key = behaviors.apply(
            lambda row: '-'.join(row.values.astype(str)), axis=1).tolist()
        behaviors.history = behaviors.history.apply(literal_eval)

        self.history = torch.from_numpy(np.array(behaviors.history.tolist()))
        if 'user' in args.dataset_attributes['behaviors']:
            self.user = torch.from_numpy(behaviors.user.to_numpy())
        if 'history_length' in args.dataset_attributes['behaviors']:
            self.history_length = torch.from_numpy(
                behaviors.history_length.to_numpy())

    def __len__(self):
        return len(self.history)

    def __getitem__(self, i):
        item = {
            'history': self.history[i],
            'key': self.key[i],
        }
        if 'user' in args.dataset_attributes['behaviors']:
            item['user'] = self.user[i]
        if 'history_length' in args.dataset_attributes['behaviors']:
            item['history_length'] = self.history_length[i]
        return item


class BehaviorsDataset(Dataset):
    """
    Load behaviors for evaluation, (user, time) pair as session
    """
    def __init__(self, behaviors_path):
        super().__init__()
        behaviors = pd.read_table(behaviors_path,
                                  usecols=args.dataset_attributes['behaviors'])

        columns = list(behaviors.columns)
        for x in ['positive_candidates', 'negative_candidates']:
            columns.remove(x)
        self.key = behaviors[columns].apply(
            lambda row: '-'.join(row.values.astype(str)), axis=1).tolist()

        behaviors.positive_candidates = behaviors.positive_candidates.apply(
            literal_eval)
        behaviors.negative_candidates = behaviors.negative_candidates.apply(
            literal_eval)

        self.positive_candidates = behaviors.positive_candidates.tolist()
        self.negative_candidates = behaviors.negative_candidates.tolist()

    def __len__(self):
        return len(self.key)

    def __getitem__(self, i):
        item = {
            'key': self.key[i],
            'positive_candidates': self.positive_candidates[i],
            'negative_candidates': self.negative_candidates[i],
        }
        return item


if __name__ == '__main__':
    from torch.utils.data import DataLoader, BatchSampler, RandomSampler

    dataset = TrainDataset(f'data/{args.dataset}/train.tsv',
                           f'data/{args.dataset}/news.tsv')
    dataloader = DataLoader(dataset,
                            sampler=BatchSampler(RandomSampler(dataset),
                                                 batch_size=args.batch_size,
                                                 drop_last=False),
                            collate_fn=lambda x: x[0],
                            pin_memory=True)
    for x in dataloader:
        print(x)
        import ipdb
        ipdb.set_trace()
        break
