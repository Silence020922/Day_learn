from __future__ import print_function

import numpy as np
# from sklearn._typing import Int
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist() # 取的是概率最高的k个label
            probs_[:] = 0
            probs_[labels] = 1
            all_labels.append(probs_)
        return np.asarray(all_labels)


class Classifier(object): # OVR下的分类器

    def __init__(self, embeddings, clf):
        self.embeddings = embeddings
        self.clf = TopKRanker(clf)
        self.binarizer = MultiLabelBinarizer(sparse_output=True) # 多标签任务label生成

    def train(self, X, Y, Y_all): 
        self.binarizer.fit(Y_all) # 提取所有数据标签类别
        X_train = [self.embeddings[x] for x in X]
        Y = self.binarizer.transform(Y)
        self.clf.fit(X_train, Y)

    def evaluate(self, X, Y):
        top_k_list = [len(l) for l in Y]
        Y_ = self.predict(X, top_k_list)
        Y = self.binarizer.transform(Y)
        averages = ["micro", "macro", "samples", "weighted"]
        results = {}
        for average in averages:
            results[average] = f1_score(Y, Y_, average=average)
        results['acc'] = accuracy_score(Y, Y_)
        print('-------------------')
        print(results)
        return results #F1-score + accuracy

    def predict(self, X, top_k_list): # top_k_list 是个什么玩意，全1呗
        X_ = np.asarray([self.embeddings[x] for x in X])
        Y = self.clf.predict(X_, top_k_list=top_k_list)
        return Y

    def split_train_evaluate(self, X, Y, train_radio, seed=0): # 训练集划分器
        """
        input:
        X: feature
        Y: label
        trian_radio: 训练比例
        seed: 设置随机种子
        """
        state = np.random.get_state() # 记录当前随机状态
        training_size = int(train_radio * len(X)) # 训练集大小
        np.random.seed(seed) # 设置随机种子
        shuffle_indices = np.random.permutation(len(X)) # (np.arange(len(X))) 等价的 相当于对X进行重排序。 
        X_train = [X[shuffle_indices[i]] for i in range(training_size)]
        Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
        X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))] # 取剩下的作为测试集
        Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

        self.train(X_train, Y_train, Y) # 训练函数
        np.random.set_state(state) # 
        return self.evaluate(X_test, Y_test) # 输出对测试集测试结果的评价


def read_node_label(filename, skip_head=False): # 有点鸡肋的函数啊。读取点的label，结构为每一行存储 节点+label
    df = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            df.readline()
        l = df.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    df.close()
    return X, Y
