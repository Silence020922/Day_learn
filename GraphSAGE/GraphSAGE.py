import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

"""
GraphSAGE包含两个关键过程,采样+聚合
"""


# 定义邻居采样函数
def sampling(src_node, sample_num, neighbor_table):
    """
    src_node 需要采样的源节点全体 list
    sample_num 采样邻居数 int
    neighbor_table 节点对应邻居信息 list or dict
    """
    results = []
    for v in src_node:
        res = np.random.choice(neighbor_table[v], sample_num)  # 有放回进行选取
        results.append(res)
    return np.asarray(results).flatten()


# 多阶采样函数
def multi_sampling(src_node, sample_num, neighbor_table):
    """
    src_node 同上定义
    sample_num 这里为每一阶所需要采样的个数 list
    neighbor_table 节点到邻居的映射
    """
    sampling_result = [src_node]
    for k, hope_k in enumerate(sample_num):  # k为阶 hope_k为k阶采样数
        hopk_result = sampling(sampling_result[k], hope_k, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result


# 邻居聚合
class NeihborAggregator(nn.Module):  # 基类
    def __init__(self, input_dim, output_dim, use_bias=False, aggr_method="mean"):
        """聚合节点邻居
        input_dim 输入特征维度 int
        output_dim 输出特征维度 int
        use_bias 是否启用偏置 default False
        aggr_method 聚合方式 default mean
        """
        super(NeihborAggregator, self).__init__()  # 调用父类初始化方法
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_baise = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))  # 生成张量W并参数化
        if self.use_baise:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))  # 生成偏置b并参数化
        self.reset_parameter()

    def reset_parameter(self):
        init.kaiming_uniform_(self.weight)  # 初始化权重使用kaiming均匀分布
        if self.use_baise:
            init.zeros_(self.bias)  # 对偏置初始0

    def forward(self, neighbor_feature):  # 前馈过程
        """
        输入的neighbor_feature 维度为 N_src*N_nei*input_dim
        weight维度为 input_dim*output_dim
        arrg_neighbor 维度为 N_src*input_dim
        """
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor = neighbor_feature.max(dim=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        else:
            raise ValueError(
                "Unknown aggr type, expected sum|max|mean,but got {0}".format(
                    self.aggr_method
                )
            )
        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)  # XW
        if self.use_baise:
            neighbor_hidden += self.bias
        return neighbor_hidden

    def extra_repr(self):
        return "in_features={}, out_features={}, aggr_method={}".format(
            self.input_dim, self.output_dim, self.aggr_method
        )


class SageGCN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        activation=F.relu,
        aggr_neighbor_method="mean",
        aggr_hidden_method="sum",
    ):
        super(SageGCN, self).__init__()
        assert aggr_neighbor_method in [
            "mean",
            "max",
            "sum",
        ]  # 限制aggr_neighbor_method选择
        assert aggr_hidden_method in ["sum", "concat"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor = aggr_neighbor_method
        self.aggr_hidden = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeihborAggregator(
            input_dim, hidden_dim, aggr_method=aggr_neighbor_method
        )
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.reset_paramater()

    def reset_paramater(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features):
        neighbor_hidden = self.aggregator(neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.weight)

        if self.aggr_hidden == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)  # 执行拼接操作
        else:
            raise ValueError(
                "Unknown aggr type, expected sum|concat,but got {0}".format(
                    self.aggr_hidden
                )
            )

        if self.activation:  # 激活函数
            return self.activation(hidden)
        else:
            return hidden

    def extra_repr(self):
        output_dim = (
            self.hidden_dim if self.aggr_hidden == "sum" else self.hidden_dim * 2
        )
        return "in_features={}, out_features={}, aggr_hidden_method={}".format(
            self.input_dim, output_dim, self.aggr_hidden
        )


# 定义GraphSage模型
class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_neighbors_list):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors_list = num_neighbors_list
        self.num_layers = len(num_neighbors_list)  # 采样阶数
        self.gcn = nn.ModuleList()
        self.gcn.append(
            SageGCN(input_dim, hidden_dim[0])
        )  # 默认激活函数RELU 均值聚合邻居特征 对隐层得到特征进行加和操作
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index + 1]))
        self.gcn.append(SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None))

    def forward(self, node_features_list):
        """
        node_features_list是一个列表,其中第0个元素代表源节点的特征,其后的元素表示每阶采样得到的节点特征。
        """
        hidden = node_features_list
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1].view(
                    (src_node_num, self.num_neighbors_list[hop], -1)
                )
                h = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]

    def extra_repr(self):
        return "in_features={}, num_neighbors_list={}".format(
            self.input_dim, self.num_neighbors_list
        )