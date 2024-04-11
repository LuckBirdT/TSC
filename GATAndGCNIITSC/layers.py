import math
import torch.nn as nn
from torch.nn.parameter import Parameter
from  data import  *
import torch.nn.functional as F


# spGAT Layer
class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class mulGAT(nn.Module):
    def __init__(self, args,nfeat,hid):
        """Sparse version of GAT."""
        super(mulGAT, self).__init__()
        self.dropout = args.dropout
        self.nhid = hid
        self.alpha = args.al
        self.nheads = args.head

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 self.nhid,
                                                 dropout=self.dropout,
                                                 alpha=self.alpha,
                                                 concat=True) for _ in range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self,x,adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        return F.elu(x)

class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj._indices()

        h = torch.mm(input, self.W)
        # h: N x out
        h = torch.where(torch.isnan(h), torch.tensor(0.0).to(h.device), h)

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        edge_e = torch.where(torch.isnan(edge_e), torch.tensor(0.0).to(edge_e.device), edge_e)
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        h_prime = torch.where(torch.isnan(h_prime), torch.tensor(0.0).to(h_prime.device), h_prime)
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        h_prime = torch.where(torch.isnan(h_prime), torch.tensor(0.0).to(h_prime.device), h_prime)

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'




################################ mask Layer
class MaskLayer(nn.Module):
    def __init__(self):
        super(MaskLayer, self).__init__()

    def get_mask(self,x, drop_prob):
        drop_mask = torch.empty(
            (x.size(1),),
            dtype=torch.float32,
            device=x.device).uniform_(0, 1) < drop_prob
        adding=torch.zeros_like(drop_mask)
        adding[-1]=True
        random.shuffle(adding)
        return drop_mask+adding

    def forward(self, last_layer,current_layer,  rate,l): # l层数
        if l <= 2:
            rate = 0
        else:
            rate = 1 - math.log(rate / (l + 1) + 1)
        # 部分卷积
        drop_mask = self.get_mask(last_layer, rate)
        hi = current_layer.clone()
        hi[:, drop_mask] = 0
        input = last_layer.clone()
        input[:, ~drop_mask] = 0
        return hi + input


###########################   GCNII
class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features,  variant=False,residual=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        #系数
        theta = math.log(lamda/l+1)
        #输入与邻接矩阵进行乘积
        hi = torch.spmm(adj, input)

        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            # 残差连接
            support = (1-alpha)*hi+alpha*h0
            # support = hi
            r = support
        output =theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output


