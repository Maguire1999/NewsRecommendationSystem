from elasticsearch import Elasticsearch as ES
import pandas as pd
import csv
from tqdm import tqdm
from espy import *
import espy
import random
import numpy as np
import json
import os
from src.espy import *

news_dataset_name = 'news_small2'
behavior_dataset_name = 'behavior_small'


es_host = '10.181.58.49:9200'


def get_user(es,user_index,user_id):
    res=es.get(index=user_index,id=user_id)
    return res
def get_user_info(user_id = 'U80234'):
    # es_host="10.181.121.227:9200"
    es=connect(es_host)
    user_inex = behavior_dataset_name
    user_dict = get_user(es,user_inex, user_id = user_id)
    return user_dict['_source']

def get_user_news_num(user_id = 'U80234',mode = 'rec'):
    name_list = get_user_info(user_id)[mode].split()
    while ' ' in name_list:
        name_list.remove(' ')
    return len(name_list)

def read_news_all():
    pass
    # user_dict = get_user_info(user_id = '')


def search_news_by_ind(ind):
    # es_host="10.181.121.227:9200"
    es=connect(es_host)
    # term: 查询 xx = “xx”
    body = {
        "query":{
            "term":{
                "ind":ind
            }
        }
    }
    res = es.search(index=news_dataset_name,doc_type="text",body=body)
    news_id = res['hits']['hits'][0]['_source']['id']
    return news_id

def search_news_by_ind2(ind):
    # es_host="10.181.121.227:9200"
    es=connect(es_host)
    body = {
        "query": {
            "ind": {
                "type": "text",
                "values":ind
            }
        }
    }
    # 搜索出id为1或2的所有数据
    res = es.search(index=news_dataset_name, doc_type="text", body=body)

def info_add(ori = "",change = [],lim = 50):
    ori_list = ori.split(" ")
    while '' in ori_list:
        ori_list.remove('')
    for item in change:
        ori_list.append(item)
    if len(ori_list) > lim:
        ori_list = ori_list[len(ori_list) - lim :]
    return " ".join(ori_list)
def update_user(user_index = 'behavior_small',user_id = 'U80234',mode = 'clicked',change_list = []):
    """
    user_index es数据库表名
    user_id 目标用户
    mode 修改的类型
    changelist 添加的内容
    """
    # es_host="10.181.121.227:9200"
    es = connect(es_host)
    user_dict = get_user_info(user_id)
    if mode == 'clicked_news':
        results = info_add(user_dict[mode],change_list,lim = 50)
        user_dict[mode] = results
    elif mode == 'impressions':
        # results = info_add(user_dict[mode], change_list, lim=20)
        user_dict[mode] = " ".join(change_list)
    elif mode == 'rec':
        results = info_add(user_dict[mode],change_list,lim=18 )
        user_dict[mode] = results

    es.update(index=user_index, doc_type="text", id=user_id, body={"doc":user_dict})

def pop_rec(user_id,pop_str):
    # es_host="10.181.121.227:9200"
    es = connect(es_host)
    user_dict = get_user_info(user_id)
    rec = user_dict['rec']
    rec = rec.replace(pop_str, '')
    if rec == ' ' or rec == " ":
        rec = ''
    user_index = behavior_dataset_name
    user_dict['rec'] = rec
    es.update(index=user_index, doc_type="text", id=user_id, body={"doc":user_dict})

#! 这里的问题 不知道是不是由于ind的不存在
def create_candidate(num = 20,user_id = ''):
    # 随机读取num个newsid
    # 问题在于如何得到全集查询到全集，而且要去除历史数据
    candidate_list = []
    max_ind = 42414
    # 总的新闻条数
    ind_rand = np.random.randint(max_ind, size=num)
    for i in ind_rand:
        candidate_list.append(search_news_by_ind(i))
    update_user(user_index =behavior_dataset_name, user_id = user_id, mode = 'impressions', change_list = candidate_list)


def get_user_clicked(user_id = 'U80234',user_name = '马永嘉'):
    # es_host="10.181.121.227:9200"
    es=connect(es_host)
    res=es.get(index=behavior_dataset_name,id=user_id)
    clicked=res["_source"]["clicked_news"].split(" ")
    while "" in clicked:
        clicked.remove("")
    category=dict()
    for cli in clicked:
        res=es.get(index="news_small2", id=cli)["_source"]["category"]
        category[res]=category.get(res,0)+1

    path = '../static/' + user_name + '.json'
    if not os.path.exists(path):
        path = './static/' + user_name + '.json'
    category_list = []
    for item in category.items():
        temp_dict = {}
        temp_dict["value"] = item[1]
        temp_dict["name"] = item[0]
        category_list.append(temp_dict)



    category_list.append(category)
    with open(path,"w") as f:
        json.dump(category_list,f)
        print("历史数据统计加载入文件完成...")

def replace_clicked(behaviors,tar_user = 'U64099',clicked_all = "",candidate_new = ""):
    # pandas通过loc设置更新对象的元素
    target_ind = behaviors[behaviors.user == tar_user].index
    behaviors.loc[target_ind,'clicked_news'] = clicked_all
    behaviors.loc[target_ind,'impressions'] = candidate_new


    return behaviors


if __name__ == '__main__':
    user_id = 'U60458'
    # usr_dict = get_user_info(user_id=user_id)
    # print(usr_dict)
    # print(usr_dict['rec'])
    print(get_user_info(user_id=user_id)['rec'])
    print("test statics")
    # get_user_clicked()
    # rec = ['N17161']
    # update_user(behavior_dataset_name,user_id,'rec',rec)
    # for rec_i in rec:
        # pop_rec(user_id,rec_i)
    print(get_user_info(user_id=user_id)['rec'])