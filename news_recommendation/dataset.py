import pandas as pd
import numpy as np
import importlib
import torch
import random
import os
import hashlib
import pickle

from pathlib import Path
from ast import literal_eval
from torch.utils.data import Dataset
from news_recommendation.shared import args, logger


class TrainDataset(Dataset):
    def __init__(self, behaviors_path, news_path, epoch=1):
        super().__init__()
        assert all(
            attribute in ['category', 'subcategory', 'title', 'abstract']
            for attribute in args.dataset_attributes['news'])
        assert all(attribute in [
            'user', 'history', 'history_length', 'positive_candidate',
            'negative_candidates'
        ] for attribute in args.dataset_attributes['behaviors'])

        pickle_path = os.path.join(
            args.cache_path,
            f"{hashlib.md5(str(args.__dict__).encode('utf-8')).hexdigest()}-{epoch}.pkl"
        )
        if args.cache_dataset and os.path.isfile(pickle_path):
            with open(pickle_path, 'rb') as f:
                logger.info(f'Load dataset cache from {pickle_path}')
                self.behaviors, self.news_pattern, self.behaviors_pattern = pickle.load(
                    f)

        else:
            self.news, self.news_pattern = self._process_news(news_path)
            self.single_news_length = list(self.news_pattern.values())[-1][-1]
            self.behaviors, self.behaviors_pattern = self._process_behaviors(
                behaviors_path)
            if args.cache_dataset:
                logger.info(f'Save dataset cache to {pickle_path}')
                Path(args.cache_path).mkdir(parents=True, exist_ok=True)
                with open(pickle_path, 'wb') as f:
                    pickle.dump([
                        self.behaviors, self.news_pattern,
                        self.behaviors_pattern
                    ], f)

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, i):
        return self.behaviors[i]

    def _process_news(self, news_path):
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
        behaviors.rename(columns={'positive_candidates': 'positive_candidate'},
                         inplace=True)
        behaviors.positive_candidate = behaviors.positive_candidate.infer_objects(
        )

        def sample_negatives(negatives):
            if len(negatives) > args.negative_sampling_ratio:
                return random.sample(negatives, args.negative_sampling_ratio)
            else:
                return negatives + [0] * (args.negative_sampling_ratio -
                                          len(negatives))

        behaviors.negative_candidates = behaviors.negative_candidates.apply(
            sample_negatives)

        behaviors_attributes2length = {
            'user':
            1,
            'history':
            args.num_history * self.single_news_length,
            'history_length':
            1,
            'positive_candidate':
            self.single_news_length,
            'negative_candidates':
            args.negative_sampling_ratio * self.single_news_length,
        }
        behaviors_elements = []
        behaviors_pattern = {}
        current_length = 0

        for attribute in args.dataset_attributes['behaviors']:
            if attribute in ['user', 'history_length', 'positive_candidate']:
                numpy_array = behaviors[attribute].to_numpy()[..., np.newaxis]
            elif attribute in ['history', 'negative_candidates']:
                numpy_array = np.array(behaviors[attribute].tolist())
            else:
                raise ValueError

            if attribute in [
                    'history', 'positive_candidate', 'negative_candidates'
            ]:
                numpy_array = self.news[numpy_array]
                numpy_array = numpy_array.reshape((numpy_array.shape[0], -1))

            behaviors_elements.append(numpy_array)
            behaviors_pattern[attribute] = (
                current_length,
                current_length + behaviors_attributes2length[attribute])
            current_length += behaviors_attributes2length[attribute]

        behaviors = np.concatenate(behaviors_elements, axis=1)
        return behaviors, behaviors_pattern


# class NewsDataset(Dataset):
#     """
#     Load news for evaluation.
#     """
#     def __init__(self, news_path):
#         super().__init__()
#         self.news_parsed = pd.read_table(
#             news_path,
#             usecols=['id'] + args.dataset_attributes['news'],
#             converters={
#                 attribute: literal_eval
#                 for attribute in set(args.dataset_attributes['news'])
#                 & set(['title', 'abstract'])
#             })
#         self.news2dict = self.news_parsed.to_dict('index')
#         for key1 in self.news2dict.keys():
#             for key2 in self.news2dict[key1].keys():
#                 if type(self.news2dict[key1][key2]) != str:
#                     self.news2dict[key1][key2] = torch.tensor(
#                         self.news2dict[key1][key2])

#     def __len__(self):
#         return len(self.news_parsed)

#     def __getitem__(self, idx):
#         item = self.news2dict[idx]
#         return item

# class UserDataset(Dataset):
#     """
#     Load users for evaluation, duplicated rows will be dropped
#     """
#     def __init__(self, behaviors_path, user2int_path):
#         super().__init__()
#         self.behaviors = pd.read_table(behaviors_path,
#                                        header=None,
#                                        usecols=[1, 3],
#                                        names=['user', 'clicked_news'])
#         self.behaviors.clicked_news.fillna(' ', inplace=True)
#         self.behaviors.drop_duplicates(inplace=True)
#         user2int = dict(pd.read_table(user2int_path).values.tolist())
#         user_total = 0
#         user_missed = 0
#         for row in self.behaviors.itertuples():
#             user_total += 1
#             if row.user in user2int:
#                 self.behaviors.at[row.Index, 'user'] = user2int[row.user]
#             else:
#                 user_missed += 1
#                 self.behaviors.at[row.Index, 'user'] = 0
#         if model == 'LSTUR':
#             print(f'User miss rate: {user_missed/user_total:.4f}')

#     def __len__(self):
#         return len(self.behaviors)

#     def __getitem__(self, idx):
#         row = self.behaviors.iloc[idx]
#         item = {
#             "user": row.user,
#             "clicked_news_string": row.clicked_news,
#             "clicked_news":
#             row.clicked_news.split()[:args.num_history]
#         }
#         item['clicked_news_length'] = len(item["clicked_news"])
#         repeated_times = args.num_history - len(
#             item["clicked_news"])
#         assert repeated_times >= 0
#         item["clicked_news"] = ['PADDED_NEWS'
#                                 ] * repeated_times + item["clicked_news"]

#         return item

# class BehaviorsDataset(Dataset):
#     """
#     Load behaviors for evaluation, (user, time) pair as session
#     """
#     def __init__(self, behaviors_path):
#         super().__init__()
#         self.behaviors = pd.read_table(behaviors_path,
#                                        header=None,
#                                        usecols=range(5),
#                                        names=[
#                                            'impression_id', 'user', 'time',
#                                            'clicked_news', 'impressions'
#                                        ])
#         self.behaviors.clicked_news.fillna(' ', inplace=True)
#         self.behaviors.impressions = self.behaviors.impressions.str.split()

#     def __len__(self):
#         return len(self.behaviors)

#     def __getitem__(self, idx):
#         row = self.behaviors.iloc[idx]
#         item = {
#             "impression_id": row.impression_id,
#             "user": row.user,
#             "time": row.time,
#             "clicked_news_string": row.clicked_news,
#             "impressions": row.impressions
#         }
#         return item

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = TrainDataset(f'data/{args.dataset}/train.tsv',
                           f'data/{args.dataset}/news.tsv')
    dataloader = DataLoader(dataset,
                            batch_size=512,
                            shuffle=True,
                            num_workers=4)
    for x in dataloader:
        print(x)
        import ipdb
        ipdb.set_trace()
        break
