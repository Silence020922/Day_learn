# pytorch 版本
import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.nn.parameter import Parameter
import math

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self,feature,input,adj,alpha,lamda,l): # l当前层
        beta = math.log(lamda/l + 1)
        if self.variant:
            hi = torch.spmm(adj,input)
            support = torch.cat([hi,feature],dim=1)
            output = beta*torch.mm(support, self.weight)+(1-beta)*((1-alpha)*hi+alpha*feature)
        else: # GCNII
            IRC = (1 - alpha)*torch.spmm(adj,input) + alpha*feature # Initial residual connection
            output = (1-beta)*IRC + beta*torch.mm(IRC,self.weight)
        if self.residual:
            output = 0.5*output + 0.5*input
        return output

class GraphConvolution2(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution2, self).__init__() 
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self,feature,input,adj,alpha): # l当前层
            output = (1 - alpha)*torch.spmm(adj,input) + alpha*feature # APPNP
            return output