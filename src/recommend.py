import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
from config import model_name
from torch.utils.data import Dataset, DataLoader
from os import path
import sys
import pandas as pd
from ast import literal_eval
import importlib
from multiprocessing import Pool
import os
from multiprocessing import Process, Pool

from utils import *
from espy import *

from src.utils import *
from src.espy import *

try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

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


class NewsDataset(Dataset):
    """
    Load news for evaluation.
    """
    def __init__(self, news_path):
        super(NewsDataset, self).__init__()
        self.news_parsed = pd.read_table(
            news_path,
            usecols=['id'] + config.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes['news']) & set([
                    'title', 'abstract', 'title_entities', 'abstract_entities'
                ])
            })
        self.news2dict = self.news_parsed.to_dict('index')
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                if type(self.news2dict[key1][key2]) != str:
                    self.news2dict[key1][key2] = torch.tensor(
                        self.news2dict[key1][key2])

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        item = self.news2dict[idx]
        return item


class UserDataset(Dataset):
    """
    Load users for evaluation, duplicated rows will be dropped
    """
    def __init__(self, behaviors_path, user2int_path ,target_user_id = 'U80234'):
        super(UserDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=[1, 3],
                                       names=['user', 'clicked_news'])
        # print(self.behaviors[self.behaviors.user == target_user_id])
        print(self.behaviors[self.behaviors.user == target_user_id].iloc[[0]])
        self.behaviors = self.behaviors[self.behaviors.user == target_user_id].iloc[[0]]
        try :
            user_dict = get_user_info(target_user_id)
            target_ind = self.behaviors[self.behaviors.user == target_user_id].index
            self.behaviors.loc[target_ind, 'clicked_news'] = user_dict['clicked_news']
        # self.behaviors.loc[target_ind, 'impressions'] = user_dict['impressions']
        except:
            print("链接数据库失败")
        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.drop_duplicates(inplace=True)
        user2int = dict(pd.read_table(user2int_path).values.tolist())
        user_total = 0
        user_missed = 0
        for row in self.behaviors.itertuples():
            user_total += 1
            if row.user in user2int:
                self.behaviors.at[row.Index, 'user'] = user2int[row.user]
            else:
                user_missed += 1
                self.behaviors.at[row.Index, 'user'] = 0
        if model_name == 'LSTUR':
            print(f'User miss rate: {user_missed/user_total:.4f}')

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "user":
            row.user,
            "clicked_news_string":
            row.clicked_news,
            "clicked_news":
            row.clicked_news.split()[:config.num_clicked_news_a_user]
        }
        item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - len(
            item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = ['PADDED_NEWS'
                                ] * repeated_times + item["clicked_news"]

        return item


class BehaviorsDataset(Dataset):
    """
    Load behaviors for evaluation, (user, time) pair as session
    """
    def __init__(self, behaviors_path,target_user_id = 'U80234'):
        super(BehaviorsDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=range(5),
                                       names=[
                                           'impression_id', 'user', 'time',
                                           'clicked_news', 'impressions'
                                       ])
        self.behaviors = self.behaviors[self.behaviors.user == target_user_id].iloc[[0]]
        try :
            user_dict = get_user_info(target_user_id)
            target_ind = self.behaviors[self.behaviors.user == target_user_id].index
            self.behaviors.loc[target_ind, 'clicked_news'] = user_dict['clicked_news']
            self.behaviors.loc[target_ind, 'impressions'] = user_dict['impressions']
        except:
            print("链接数据库失败")
        self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors.impressions = self.behaviors.impressions.str.split()

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "impression_id": row.impression_id,
            "user": row.user,
            "time": row.time,
            "clicked_news_string": row.clicked_news,
            "impressions": row.impressions
        }
        return item


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
def evaluate(model, directory, num_workers, max_count=sys.maxsize , target_user_id = 'U80234'):
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
    news2vector_path = directory + '/news2vector.pt'
    print(os.path.exists(news2vector_path))
    if os.path.exists(news2vector_path):
        print("news2vector exist")
        news2vector = torch.load(news2vector_path)
        try :
            news2vector = news2vector.to(device)
        except:
            pass

    else:
        news_dataset = NewsDataset(path.join(directory, 'news_parsed.tsv'))
        news_dataloader = DataLoader(news_dataset,
                                     batch_size=config.batch_size * 16,
                                     shuffle=False,
                                     num_workers=config.num_workers,
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
        torch.save(news2vector,news2vector_path)
        print("success save ")

    user2int_path = '../data/train/user2int.tsv'
    if os.path.exists(user2int_path):
        user_dataset = UserDataset(path.join(directory, 'behaviors.tsv'),
                               '../data/train/user2int.tsv',target_user_id)
    else:
        user_dataset = UserDataset(path.join(directory, 'behaviors.tsv'),
                               './data/train/user2int.tsv',target_user_id)
    # target_user = user_dataset.ix(target_user_id)
    # print(target_user)
    user_dataloader = DataLoader(user_dataset,
                                 batch_size=1,# config.batch_size * 16,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=True)

    user2vector = {}
    for minibatch in tqdm(user_dataloader,
                          desc="Calculating vectors for users"):
        user_strings = minibatch["clicked_news_string"]
        if any(user_string not in user2vector for user_string in user_strings):
            clicked_news_vector = torch.stack([
                torch.stack([news2vector[x].to(device) for x in news_list],
                            dim=0) for news_list in minibatch["clicked_news"]
            ],
                                              dim=0).transpose(0, 1)
            if model_name == 'LSTUR':
                user_vector = model.get_user_vector(
                    minibatch['user'], minibatch['clicked_news_length'],
                    clicked_news_vector)
            else:
                user_vector = model.get_user_vector(clicked_news_vector)
            for user, vector in zip(user_strings, user_vector):
                if user not in user2vector:
                    user2vector[user] = vector

    behaviors_dataset = BehaviorsDataset(path.join(directory, 'behaviors.tsv'),target_user_id)


    behaviors_dataloader = DataLoader(behaviors_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=config.num_workers)

    count = 0

    tasks = []

    for minibatch in tqdm(behaviors_dataloader,
                          desc="Calculating probabilities"):
        count += 1
        if count == max_count:
            break
        # for news in minibatch['impressions']:
        #     print(news)
        try:
            # print(minibatch['impressions'][0][0])
            test = 'N46481'
            candidate_news_vector = torch.stack([news2vector[news[0]]
                 for news in minibatch['impressions']], dim=0)
            # candidate_news_vector = torch.stack([
            #     news2vector[test]], dim=0)
        except:
            candidate_news_vector = torch.stack([
                news2vector[news[0].split('-')[0]]
                for news in minibatch['impressions']
            ],dim=0)
        user_vector = user2vector[minibatch['clicked_news_string'][0]]
        click_probability = model.get_prediction(candidate_news_vector,
                                                 user_vector)

        y_pred = click_probability.tolist()
        # y_pred = torch.softmax(torch.tensor(y_pred)).tolist()
        try:
            # y_true = [
            #     int(news[0].split('-')[1]) for news in minibatch['impressions']
            # ]
            candidate_news_id_ = [news2vector[news[0]]
                for news in minibatch['impressions']]
            # candidate_news_id = ['N46481']
            candidate_news_id = [news[0] for news in minibatch['impressions']]
        except:
            # y_true = [
            #     int(news[0].split('-')[1]) for news in minibatch['impressions']
            # ]
            candidate_news_id = [news[0].split('-')[0] for news in minibatch['impressions']]
        # tasks.append((y_true, y_pred))

    def softmax(x):
        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x)
        y = exp_x / sum_exp_x

        return y
    y_pred = [(item+1)/2 for item in y_pred]
    rec_sort_index = np.argsort(-np.array(y_pred))
    return np.array(candidate_news_id)[rec_sort_index],np.array(y_pred)[rec_sort_index]

    #
    # with Pool(processes=num_workers) as pool:
    #     results = pool.map(calculate_single_user_metric, tasks)

    # aucs, mrrs, ndcg5s, ndcg10s = np.array(results).T
    # return np.nanmean(aucs), np.nanmean(mrrs), np.nanmean(ndcg5s), np.nanmean(
    #     ndcg10s)


def recommend(user_id = 'U80234',thr = 0.5):
    create_candidate(num=20, user_id=user_id)
    root = './data/test'
    if os.path.exists(root):
        pass
    else:
        root = '../data/test'
    checkpoint_path = './checkpoint'
    if os.path.exists(checkpoint_path):
        pass
    else:
        checkpoint_path = '../checkpoint'
    print('Using device:', device)
    print(f'Evaluating model {model_name}')
    # Don't need to load pretrained word/entity/context embedding
    # since it will be loaded from checkpoint later
    model = Model(config).to(device)
    from train import latest_checkpoint  # Avoid circular imports
    checkpoint_path = latest_checkpoint(path.join(checkpoint_path, model_name))
    if checkpoint_path is None:
        print('No checkpoint file found!')
        exit()
    print(f"Load saved parameters in {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path)
    except:
        checkpoint = torch.load(checkpoint_path,map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    rec_list,rec_prob = evaluate(model,root ,config.num_workers,target_user_id=user_id)
    print("推荐新闻ID：",rec_list)
    print("推荐度：",rec_prob)
    update_user('behavior_small', user_id, 'rec', rec_list[rec_prob>thr])
    return rec_list


def rec_pool(user_id='U80234'):
    if get_user_news_num(user_id) < 10:
        recommend(user_id=user_id, thr=0.5)


if __name__ == '__main__':
    users = ["马永嘉", "叶竹鹏", "王开宇"]
    user_id_dict = {users[0]: 'U80234', users[1]: 'U60458', users[2]: 'U44190', 'guest': 'U9444'}
    pool = Pool(4)
    for i in range(3):
        rec_pool(user_id_dict[users[i]])
        # pool.apply_async(rec_pool,args=(user_id_dict[users[i]],) )
    # pool.close()
    # pool.join()
    # user_id = 'U80234'
    # rec_pool(user_id)
    # try:
    #     rec_pool(user_id)
    # except:
    #     print("没有推荐")
    #     pass