from elasticsearch import Elasticsearch as ES
import pandas as pd
import csv
from tqdm import tqdm

news_dataset_name = 'news_small2'

es_host = '10.181.58.49:9200'
def connect(es_host):
    es=ES(es_host)
    es = ES([es_host],
                    # 在做任何操作之前，先进行嗅探
                    # sniff_on_start=True,
                    # 节点没有响应时，进行刷新，重新连接
                    # sniff_on_connection_fail=True,
                    # # 每 60 秒刷新一次
                    # sniffer_timeout=60
                    )

                    ###########################关于基本信息的查看############
    # #测试是否能连通
    # print(f"测试连通性：{es.ping()}")
    return es

def upload_news(dir_news):
    news_datas= pd.read_table(dir_news,
                         header=None,
                         usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                         quoting=csv.QUOTE_NONE,
                         names=[
                             'id', 'category', 'subcategory', 'title',
                             'abstract','address', 'title_entities', 'abstract_entities'
                         ])  # TODO try to avoid csv.QUOTE_NONE
    news_datas.title_entities.fillna('[]', inplace=True)
    news_datas.abstract_entities.fillna('[]', inplace=True)
    news_datas.fillna(' ', inplace=True)

    index= news_dataset_name
    doc_type="text"
    i=0
    for news in tqdm(news_datas.itertuples(),desc="upload news..."):
        if i>=98888:
            news_id=news.id
            doc=dict()
            doc['id']=news.id
            doc['category']=news.category
            doc['subcategory']=news.subcategory
            doc['title']=news.title
            doc['abstract']=news.abstract
            doc['address']=news.address
            doc['title_entities']=news.title_entities
            doc['abstract_entities']=news.abstract_entities
            es.create(index=index,id=news_id,doc_type=doc_type,document=doc)
        i+=1

def get_news(es,news_index,news_id):
    res=es.get(index=news_index,id=news_id)
    return res
def get_news_info(news_id = 'N5771'):
    # es_host="10.181.121.227:9200"
    es=connect(es_host)
    news_dict = get_news(es, news_dataset_name, news_id)
    return news_dict['_source']

#! 这里的问题 不知道是不是由于ind的不存在
def get_news_by_ind(ind = 1):
    # es_host="10.181.121.227:9200"
    es=connect(es_host)
    res = es.get(index=news_dataset_name, ind = ind)['_source']
    return res['id']

if __name__ == '__main__':
    print(get_news_info())
