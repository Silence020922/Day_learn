import networkx as nx
import numpy as np
from utils import pre_nxgraph
from utils import partition_dict
from utils import partition_list
from classify import read_node_label, Classifier
from sklearn.preprocessing import MultiLabelBinarizer


# # Create a graph
# graph = nx.fast_gnp_random_graph(n=20, p=0.2)

# # pre_nxgraph
# a,b = pre_nxgraph(graph)

# #
# dict2part_list = partition_dict(b,5)
# list2part_list = partition_list(a,5)

# # read_node_label
# X_all,Y_all = read_node_label('/home/Anhao/Documents/Python/GNN/Deep_walk/data/wiki_labels.txt')
# # get_state
# # state = np.random.get_state() # 记录random当前的state 只要set相同的state就可以获得相同的打乱顺序。——保持相同的随机器


# def split_train_evaluate(X, Y, train_radio, seed=0): # 训练集划分器
#     """
#     input:
#     X: feature
#     Y: label
#     trian_radio: 训练比例
#     seed: 设置随机种子
#     """
#     state = np.random.get_state() # 记录当前随机状态
#     training_size = int(train_radio * len(X)) # 训练集大小
#     np.random.seed(seed) # 设置随机种子
#     shuffle_indices = np.random.permutation(len(X)) # (np.arange(len(X))) 等价的 相当于对X进行重排序。
#     X_train = [X[shuffle_indices[i]] for i in range(training_size)]
#     Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
#     X_test = [X[shuffle_indices[i]] for i in range(training_size, len(X))] # 取剩下的作为测试集
#     Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]
#     return X_train , Y_train, Y

# X_train, Y_train, Y = split_train_evaluate(X_all,Y_all,0.8,seed=0)
# top_k_list = [len(l) for l in Y]
# binarizer = MultiLabelBinarizer(sparse_output=True)
# binarizer.fit(Y)
# Y_train = binarizer.transform(Y_train)


def create_alias_table(area_ratio): # walk to neigh's prob
    """
    :param area_ratio: sum(area_ratio)=1
    :return: accept,alias
    """
    l = len(area_ratio) 
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio)*l # 还原到非归一化的状态
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop() 
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - (1 - area_ratio_[small_idx])
        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1
    return accept, alias


def get_alias_edge(G,t,v,p=1,q=1): # 输入 t v 具体的
    """
    compute unnormalized transition probability between nodes v and its neighbors give the previous visited node t.
    :param t:
    :param v:
    :return:
    """
    unnormalized_probs = []
    for x in G.neighbors(v): # x 为 v邻居
        weight = G[v][x].get('weight', 1.0)  # w_vx
        if x == t:  # inwalk
            unnormalized_probs.append(weight / p)
        elif G.has_edge(x, t):  # 到x 与到 t 距离相同
            unnormalized_probs.append(weight)
        else:  # outwalk
            unnormalized_probs.append(weight / q)
    norm_const = sum(unnormalized_probs)
    normalized_probs = [
        float(u_prob) / norm_const for u_prob in unnormalized_probs]
    return create_alias_table(normalized_probs)


G = nx.read_edgelist(
    "/home/Anhao/Documents/Python/GNN/Deep_walk/data/wiki_edgelist.txt",
    create_using=nx.DiGraph(),
    nodetype=None,
    data=[("weight", int)],
)  # 生成图G 但他妈的原数据没有权重

alias_nodes = {}
for node in G.nodes(): # 2405个点
    unnormalized_probs = [
        G[node][nbr].get("weight", 1.0) for nbr in G.neighbors(node)  # 对当前node所有邻居给出walk prob
    ]
    norm_const = sum(unnormalized_probs)
    normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs] # 归一化处理
    alias_nodes[node] = create_alias_table(normalized_probs)
alias_edges = {}
for edge in G.edges():
    alias_edges[edge] = get_alias_edge(G,edge[0], edge[1])
    if not G.is_directed():
        alias_edges[(edge[1], edge[0])] = get_alias_edge(edge[1], edge[0])
    alias_edges = alias_edges

alias_nodes = alias_nodes
# print(alias_nodes)
print(alias_edges)