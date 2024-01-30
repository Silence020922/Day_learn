import numpy as np
import pickle as pkl
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import networkx as nx
import tensorflow as tf

# load 文件
def load_index(filename):
    f = open(filename)
    index = []
    for line in f:
        index.append(int(line.strip()))  # 去除换行符，转化为int类型
    return index


# 随机从allx中选取训练集和测试集
def random_choice(total_n, t, v):
    """
    t: data_number of train
    v: data_number of validation
    """
    if t + v > total_n:
        print(
            "Out of range {}, final t = {}, v = {}. ".format(
                total_n, min(t, total_n), total_n - min(t, total_n)
            )
        )
    index_all = np.arange(total_n)
    np.random.shuffle(index_all)
    index_train = index_all[:t]
    index_val = index_all[t : t + v]
    return index_train, index_val


# 用来决定测试集和训练集
def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return mask.astype(bool)

# 导入数据并恢复到原来排序
def load_dataset(dataset_name,t,v = 500):
    """
    Input:
    dataset_name: 使用的数据集名称，如'cora','citeseer','pubmed'
    t: 使用训练集大小
    v: 使用测试集大小
    """
    objects = []
    names = ["tx", "ty", "allx", "ally", "graph"]
    for name in names:
        with open("data/ind.{}.{}".format(dataset_name, name), "rb") as f:
            objects.append(pkl.load(f, encoding="latin1"))
    tx, ty, allx, ally, graph = tuple(objects)  # 注意，这里graph是一个邻接表的形式
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))  # 转化为邻接矩阵csr
    test_idx = load_index("data/ind.{}.test.index".format(dataset_name))

    test_idx_sorted = np.sort(test_idx)  # 排序版本
    if dataset_name == "citeseer":
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(test_idx_sorted[0], test_idx_sorted[-1] + 1)
        tx_extend = sp.lil_matrix((len(test_idx_range_full), allx.shape[1]))  # 创建稀疏矩阵
        tx_extend[test_idx_sorted - test_idx_sorted[0], :] = tx
        tx = tx_extend
        ty_extended = np.zeros((len(test_idx_range_full), ally.shape[1]))  # 补全孤立点
        ty_extended[test_idx_sorted - test_idx_sorted[0], :] = ty
        ty = ty_extended
    features = features = sp.vstack((allx, tx)).tolil()  # 纵向拼接转化为LIL
    features[test_idx, :] = features[test_idx_sorted, :]  # 恢复排序的特征
    labels = np.vstack((ally, ty))
    labels[test_idx, :] = labels[test_idx_sorted, :]
    # 下面开始区分训练、验证、测试集
    index_tr, index_val = random_choice(allx.shape[0], t, v)  # 默认选择500个验证数据
    index_test = test_idx_sorted.tolist()
    mask_tr = sample_mask(index_tr, labels.shape[0])
    mask_val = sample_mask(index_val, labels.shape[0])
    mask_test = sample_mask(index_test, labels.shape[0])

    # 直接导出数据集
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[mask_tr,:]= labels[mask_tr, :]  # 防止多标签的情况出现，加[,:]
    y_val[mask_val,:] = labels[mask_val, :]
    y_test[mask_test,:] = labels[mask_test, :]
    return adj, features, y_train, y_val, y_test, mask_tr, mask_val, mask_test

# 求解度矩阵
def degree_matrix5(adj):
    """
    Input:
    adj: 邻接矩阵CSR
    Output:
    degree_mtx: D^(-1/2)
    """
    adj = sp.coo_matrix(adj)  # 转化为稀疏矩阵COO的形式存储
    degree_seq = np.array(adj.sum(1)).reshape(-1)  # 计算度序列
    degree_sqrt = np.power(degree_seq,-0.5)
    degree_sqrt[np.isinf(degree_sqrt)] = 0 # 除数为0的情况
    degree_mtx = sp.diags(degree_sqrt)  # 转化为对角矩阵
    return degree_mtx


# 对邻接矩阵的处理 A --> \tilde(A)
def norm_adj(adj):
    adj = adj + sp.eye(adj.shape[0]) # A+I
    d_mtx_sqrt = degree_matrix5(adj) # \tilde(D)
    return adj.dot(d_mtx_sqrt).transpose().dot(d_mtx_sqrt).tocoo() # 这里的transpose是否需要，为什么会影响精度

# 稀疏矩阵转化为tuple
def sparse_mtx_to_tutple(sparse_mtx):
    def to_tuple(mtx):
        if not sp.isspmatrix_coo(mtx):
            mtx = mtx.tocoo()  # 转化为COO形式存储的稀疏矩阵
        coords = np.vstack((mtx.row, mtx.col)).transpose() 
        values = mtx.data
        shape = mtx.shape
        return coords,values,shape # 返回值本质为多值元胞数组
    if isinstance(sparse_mtx, list):  # 是否为list type
        for i in range(len(sparse_mtx)):
            sparse_mtx[i] = to_tuple(sparse_mtx[i])
    else:
        sparse_mtx = to_tuple(sparse_mtx)
    return sparse_mtx

def preprocess_adj(adj):
    norm_Adj = norm_adj(adj) 
    return sparse_mtx_to_tutple(norm_Adj)



def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()  # 次方
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_mtx_to_tutple(features)

def dropout(X, keep_p, noise_shape, sparse=False):
    if not sparse:
        return tf.compat.v1.nn.dropout(X, keep_p)
    uniform = tf.random.uniform(shape=noise_shape)
    prob = keep_p + uniform
    dropout_mask = tf.cast(tf.floor(prob), dtype=bool)
    pre_out = tf.sparse.retain(X, dropout_mask)
    return pre_out * (1.0 / keep_p)


def dot(x, y, sparse=False):
    if sparse:
        ans = tf.sparse.sparse_dense_matmul(x, y)
    else:
        ans = tf.matmul(x, y)
    return ans