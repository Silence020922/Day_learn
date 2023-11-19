
import numpy as np

from classify import read_node_label, Classifier
from Deepwalk import DeepWalk
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE


def evaluate_embeddings(embeddings):
    X, Y = read_node_label('/home/Anhao/Documents/Python/GNN/Deep_walk/data/wiki_labels.txt') # 使用绝对路径导入数据
    tr_radio = 0.8 # 训练比例
    print("Training classifier using {:.2f}% nodes...".format(
        tr_radio * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_radio)


def plot_embeddings(embeddings,):
    X, Y = read_node_label('/home/Anhao/Documents/Python/GNN/Deep_walk/data/wiki_labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2) # 高维数据降维
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    G = nx.read_edgelist('/home/Anhao/Documents/Python/GNN/Deep_walk/data/wiki_edgelist.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)]) # 有向无权图。

    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    model.train(window_size=5, iter=3)
    embeddings = model.get_embeddings()

    evaluate_embeddings(embeddings)
    plot_embeddings(embeddings)