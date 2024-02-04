import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import networkx as nx
import torch

# copy from gcn
def parse_index_file(filename):
    f = open(filename)
    index = []
    for line in f:
        index.append(int(line.strip()))  # 去除换行符，转化为int类型
    return index

# copy from gcn 读取data中数据
def load_data(dataset_str): 
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), "rb") as f:
            if sys.version_info > (3, 0):  # python version
                objects.append(pkl.load(f, encoding="latin1"))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(
        objects
    )  # 训练集140样本，样本特征1433,测试集1000样本，allx中1708样本(除了测试集的所有样本)，7分类任务(cora)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset_str)
    )
    test_idx_range = np.sort(test_idx_reorder)  # 排序版test index
    # 找到citeseer的孤立点1000-->1015，并在对应位置用0填充
    if dataset_str == "citeseer":
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))  # 切片
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))  # 补全孤立点
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()  # allx,tx合并转化为LIL，完整的graph
    features[test_idx_reorder, :] = features[test_idx_range, :]  # 将特征还原，这样才能与图结构一一对应
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    
    # 合并还原labels
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))  # 前len(y)个作为训练集
    idx_val = range(len(y), len(y) + 500)  # 500个验证集

    return adj, features, labels, idx_train, idx_val, idx_test

def sparse_to_torch_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)  # 转化为COO形式存储的稀疏矩阵
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# feature 预处理
def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()  # 次方
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return torch.FloatTensor(np.array(features.todense())).float()

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^(-1/2) A D^(-1/2)

# adj 预处理
def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_torch_tensor(adj_normalized)

def preprocess_labels(labels):
    labels = torch.tensor(labels)
    labels = torch.argmax(labels,dim = 1)
    return labels

def acc(output,labels):
    predict = torch.argmax(output,dim=1)
    total = len(labels)
    T = predict.eq(labels)
    acc = T.sum().item()/total * 100
    return acc


