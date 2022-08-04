# save this as app.py
from flask import Flask, escape, request, render_template, redirect, Markup
from pyquery import PyQuery as pq
from urllib.parse import unquote
import random
from recommend import *
from espy import *
from utils import *
from multiprocessing import Process, Pool

from src.utils import pop_rec, update_user
from src.espy import *
from src.recommend import *

app = Flask(__name__)
user_num = 6

users = ["马永嘉", "李怡林", "王开宇","胡嘉豪","王轶群","雷昆仑"]
user_id = {users[0]: 'U80234', users[1]: 'U60458', users[2]: 'U44190', users[3]: 'U9444',users[4]: 'U9444',users[5]: 'U44190'}
global cur_user_id
cur_user_id = 'U80234'


@app.route('/')
def hello():
    return redirect('/index/马永嘉')
    # return redirect('/index/guest')


@app.route('/test', methods=["GET", "POST"])
def test():
    a = request.args.get("a")
    b = request.form.get("b", "")
    # content=random.randrange(1,100)
    content = a + "|||" + b
    return render_template("test.html", content=str(content))


@app.route('/visit/<newsid>')
def visit(newsid):
    # 历史记录保存
    # try:
    print(cur_user_id)
    user = request.args.get("user")
    print("user:", user)
    newsid_list = []
    newsid_list.append(newsid)
    update_user(user_index='behavior_small', user_id=user_id[user], mode='clicked_news', change_list=newsid_list)
    pop_rec(user_id[user], newsid)
    # pop rec
    # except:
    #     pass
    title = get_news_info(newsid)['title']
    content = "新闻内容"
    user = request.args.get("user")
    url = request.args.get("url", "")
    content = pq(unquote(url))("#main").html()
    content = "目标网站无法访问" if content is None else content
    return render_template("blog.html", users=users, title=title, content=Markup(content))


@app.route('/recommand')
def recommand(user):
    id_list = []
    return id_list


@app.route('/index')
def index_default():
    # return redirect("/index/guest")
    return redirect('/index/马永嘉')

@app.route('/index/<user>')
def index(user=""):
    newslist = []  # 缓存新闻信息
    # newid_list = recommend(user_id[user])
    # 读取该用户的推荐信息
    global cur_user_id
    cur_user_id = user_id[user]
    print(cur_user_id)
    get_user_clicked(user_id=user_id[user], user_name=user)
    # cur_user_id = user_id[user]
    try:
        usr_dict = get_user_info(cur_user_id)
        newid_list = usr_dict['rec'].split(" ")
        # while(len(newid_list)<9):
        #     rec_pool(cur_user_id)
        while "" in newid_list:
            newid_list.remove("")
        print(newid_list)
        # f = open("../data/test/news.tsv")
        num = 9
        if len(newid_list) < 9:
            num = len(newid_list)
            if num < 3:
                rec_pool(cur_user_id)
        for i in range(num):
            # l = f.readline()
            # news_tsv = l.split("\t")
            news = get_news_info(newid_list[i])
            newslist.append({
                "id": news['id'],
                "type": news['category'],
                "subtype": news['subcategory'],
                "name": news['title'],
                "summary": news['abstract'],
                "url": news['address'],
            })
    except:
        newslist = error_get_news_list()
    # for i in range(10):
    #     newslist.append({"id": str(i), "name": "新闻" + str(i), "summary": "这是新闻详情" + str(i)})
    return render_template("index.html", list=newslist, users=users)


def error_get_news_list():
    newslist = []
    f = open("../data/test/news.tsv")
    for i in range(9):
        l = f.readline()
        news = l.split("\t")
        newslist.append({
            "id": news[0],
            "type": news[1],
            "subtype": news[2],
            "name": news[3],
            "summary": news[4],
            "url": news[5],
        })
    return newslist


@app.route("/nolike/<newsid>")
def nolike(newsid):
    print("no like newsid" + newsid)
    user = request.args.get("user")
    pop_rec(user_id[user], newsid)
    return "OK"


if __name__ == '__main__':
    print("start")
    pool = Pool(4)
    for i in range(user_num):
        rec_pool(user_id[users[i]])
        # pool.apply_async(rec_pool,args=(user_id[users[i]],) )
    # pool.close()
    # pool.join()

    app.run(host="0.0.0.0")
    print("end")
