import math
import random
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from  data import  *


################################  CP-SGC
class Part_GraphConvolution(nn.Module):
    def __init__(self):
        super(Part_GraphConvolution, self).__init__()

    def get_mask(self,x, drop_prob):
        drop_mask = torch.empty(
            (x.size(1),),
            dtype=torch.float32,
            device=x.device).uniform_(0, 1) < drop_prob
        adding=torch.zeros_like(drop_mask)
        adding[-1]=True
        random.shuffle(adding)
        return drop_mask +adding

    def forward(self, input, adj, rate,l): # l层数
        if l <= 2:
            rate = 0
        else:
            rate = 1 - math.log(rate / (l + 1) + 1)
        # 部分卷积
        drop_mask = self.get_mask(input, rate)
        hi = torch.sparse.mm(adj, input)
        hi = hi.clone()
        hi[:, drop_mask] = 0
        input = input.clone()
        input[:, ~drop_mask] = 0
        return hi + input

#######################################  GCN
class GCNLayer(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
