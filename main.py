# -*-coding:utf8-*-

import math
import os
import random


class ItemBasedCF:
    def __init__(self, path):
        self.train = {}  # 用户-物品的评分表 训练集
        self.test = {}  # 用户-物品的评分表 测试集
        self.generate_dataset(path)

    def loadfile(self, path):
        with open(path, 'r', encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                yield line.strip('\r\n')

    def generate_dataset(self, path, pivot=0.7):
        # 读取文件，并生成用户-物品的评分表和测试集
        # i = 0
        for line in self.loadfile(path):
            user, movie, rating, _ = line.split('::')
            # if i <= 10:
            #     print('{},{},{},{}'.format(user, movie, rating, _))
            # i += 1
            if random.random() < pivot:
                self.train.setdefault(user, {})
                self.train[user][movie] = int(rating)
            else:
                self.test.setdefault(user, {})
                self.test[user][movie] = int(rating)

    def ItemSimilarity(self):
        # 建立物品-物品的共现矩阵
        C = dict()  # 物品-物品的共现矩阵
        N = dict()  # 物品被多少个不同用户购买
        for user, items in self.train.items():
            for i in items.keys():
                N.setdefault(i, 0)
                N[i] += 1
                C.setdefault(i, {})
                for j in items.keys():
                    if i == j:
                        continue
                    C[i].setdefault(j, 0)
                    C[i][j] += 1
        # 计算相似度矩阵
        self.W = dict()
        for i, related_items in C.items():
            self.W.setdefault(i, {})
            for j, cij in related_items.items():
                self.W[i][j] = cij / (math.sqrt(N[i] * N[j]))
        return self.W, C, N

    # 给用户user推荐，前K个相关用户
    def Recommend(self, u, K=3, N=10):
        rank = dict()
        action_item = self.train[u]  # 用户u产生过行为的item和评分
        for i, score in action_item.items():
            # j：物品j
            # wj：物品i和物品j的相似度
            for j, wj in sorted(self.W[i].items(), key=lambda x: x[1], reverse=True)[0:K]:
                if j in action_item.keys():
                    continue
                rank.setdefault(j, 0)
                # 用户u对物品j感兴趣程度：用户对物品i的打分 * 物品i和物品j的相似度
                rank[j] += score * wj
        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:N])

    # 计算召回率和准确率
    # 召回率 = 推荐的物品数 / 所有物品集合
    # 准确率 = 推荐对的数量 / 推荐总数
    def recallAndPrecision(self, k=8, nitem=10):
        hit = 0
        recall = 0
        precision = 0
        for user, items in self.test.items():
            rank = self.Recommend(user, K=k, N=nitem)
            hit += len(set(rank.keys()) & set(items.keys()))
            recall += len(items)
            precision += nitem
        return hit / (recall * 1.0), hit / (precision * 1.0)


class MovieDic:
    def __init__(self, path):
        self.movies = {}  # 电影
        self.generate_dataset(path)

    def loadfile(self, path):
        with open(path, 'r', encoding='utf-8') as fp:
            for i, line in enumerate(fp):
                yield line.strip('\r\n')

    def generate_dataset(self, path):
        # 读取文件，并生成用户-物品的评分表和测试集
        # i = 0
        for line in self.loadfile(path):
            movieId, name, tags = line.split('::')
            self.movies.setdefault(movieId, {})
            self.movies[movieId]['name'] = name
            self.movies[movieId]['tags'] = tags


path = os.path.join('ml-1m', 'ratings.dat')
icf = ItemBasedCF(path)
icf.ItemSimilarity()
recomend = icf.Recommend('3')
traied = icf.train['3']
moviePath = os.path.join('ml-1m', 'movies.dat')
movieDic = MovieDic(moviePath)
movies = movieDic.movies
print('-------------------------train----------------------------')
for k, v in traied.items():
    movieName = movies[k]['name']
    movieTags = movies[k]['tags']
    print(movieName, '----', movieTags)
print('-------------------------recommend------------------------')
for k, v in recomend.items():
    movieName = movies[k]['name']
    movieTags = movies[k]['tags']
    print(movieName, '----', movieTags)

