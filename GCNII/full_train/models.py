import torch.nn as nn
import torch.nn.functional as fun
from layers import *


class GCNII(nn.Module):
    # 全连接 卷积层 全连接
    def __init__(self,input_dim,label_class_num,layer_num,hidden_size,alpha,lamda,dropout,variant,residual):
        super(GCNII,self).__init__()
        self.convs = nn.ModuleList()
        self.liner = nn.ModuleList()
        self.act_fn = nn.ReLU()
        self.layer_num = layer_num
        self.alpha = alpha 
        self.lamda = lamda
        self.dropout = dropout
        for _ in range(layer_num):
            self.convs.append(GraphConvolution(hidden_size,hidden_size,residual,variant))
        self.liner.append(nn.Linear(input_dim,hidden_size))
        self.liner.append(nn.Linear(hidden_size,label_class_num))
        self.params1 = list(self.convs.parameters()) # 卷积过程参数
        self.params2 = list(self.liner.parameters()) # 线性变换过程参数
    def forward(self,feature,adj):
        # dropoout
        feature = fun.dropout(feature,self.dropout,training=self.training)
        # 全连接
        _layer = [] # 记录每一层输出
        _layer.append(self.act_fn(self.liner[0](feature)))
        for i in range(self.layer_num):
            input = fun.dropout(_layer[-1],self.dropout,training=self.training)
            _layer.append(self.act_fn(self.convs[i](_layer[0],input,adj,self.alpha,self.lamda,i+1)))
        input = fun.dropout(_layer[-1],self.dropout,training=self.training)
        output = self.liner[-1](input) # 经过一个全连接
        output = fun.log_softmax(output,dim=1) # log_softmax
        return output


class APPNP(nn.Module):
    # 全连接 卷积层 全连接
    def __init__(self,input_dim,label_class_num,layer_num,hidden_size,alpha,dropout):

        super(APPNP,self).__init__()
        self.layer_num = layer_num
        self.alpha = alpha 
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.liner = nn.ModuleList()
        self.act_fn = nn.ReLU()
        for _ in range(layer_num):
            self.convs.append(GraphConvolution2(hidden_size,hidden_size))
        self.liner.append(nn.Linear(input_dim,hidden_size))
        self.liner.append(nn.Linear(hidden_size,label_class_num))

        self.params1 = list(self.convs.parameters()) # 卷积过程参数
        self.params2 = list(self.liner.parameters()) # 线性变换过程参数
    def forward(self,feature,adj):
        # dropoout
        feature = fun.dropout(feature,self.dropout,training=self.training)
        # 全连接
        _layer = [] # 记录每一层输出
        _layer.append(self.act_fn(self.liner[0](feature)))
        for i in range(self.layer_num):
            input = fun.dropout(_layer[-1],self.dropout,training=self.training)
            _layer.append(self.convs[i](_layer[0],input,adj,self.alpha)) # APPNP卷积层中间没有非线性层
        input = fun.dropout(_layer[-1],self.dropout,training=self.training)
        output = self.liner[-1](input) # APPNP在最后会通过一个Relu层
        output = fun.log_softmax(output,dim=1) # log_softmax
        return output
