import torch
from layers import *
import torch.nn as nn
from  metrics import *
import torch.nn.functional as F



#2018 GAT
class GAT(nn.Module):
    def __init__(self, args,nfeat, nclass):
        """Sparse version of GAT."""
        super(GAT, self).__init__()
        self.hid = args.hid
        self.nlayer = args.nlayer


        self.attentions = [mulGAT(args,nfeat if _ == 0 else self.hid,self.hid//args.head) for _ in range(self.nlayer-1)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = mulGAT(args,self.hid,nclass)


    def forward(self, x, adj):
        for attention in self.attentions:
            x = attention(x,adj)
        x = self.out_att(x,adj)
        return F.log_softmax(x, dim=1),0,0


class GAT_TSC(nn.Module):
    def __init__(self, args,nfeat, nclass):
        """Sparse version of GAT."""
        super(GAT_TSC, self).__init__()
        self.hid = args.hid
        self.nlayer = args.nlayer

        self.attentions = [mulGAT(args,nfeat if _ == 0 else self.hid,self.hid//args.head) for _ in range(self.nlayer-1)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = mulGAT(args,self.hid,nclass)
        self.mask_layer = MaskLayer()
        self.tau = args.tau
        self.lamda = args.lamda
        self.k = args.k

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def bind_loss(self, z1,z2):  # 约束  使用距离约束是有效的说明学出来的特征之间应该是非常相近的
        mask = 1-torch.eye(z1.shape[0]).to(z1.device)
        # 对比学习约束
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        neg_c2 = refl_sim * mask
        neg_c3 = between_sim * mask
        CT = -torch.log( between_sim.diag() / (neg_c2.sum(1) + neg_c3.sum(1)) )
        return CT.mean()
    def forward(self, x, adj):
        last = x
        loss = 0
        for i,attention in enumerate(self.attentions):
            current = attention(last,adj)
            if i>0:
                current = self.mask_layer(last,current,self.lamda,i+2)
                if i%self.k ==0:
                    loss += self.bind_loss(current,current)
            last = current

        x = self.out_att(last,adj)
        return F.log_softmax(x, dim=1),loss,0,0


#2020  cora nlayer 64
class GCNII_TSC(nn.Module):
    def __init__(self, args,nfeat,  nclass ):
        super(GCNII_TSC, self).__init__()
        #nlayers, nhidden,dropout, lamda, alpha, variant
        nlayers=args.nlayer
        nhidden=args.hid

        self.convs = nn.ModuleList()
        for _ in range(nlayers): #默认层数为64层 但是发现，在四五层效果也比较不错了
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=args.variant))

        self.fcs = nn.ModuleList()
        #线性变换
        self.fcs.append(nn.Linear(nfeat, nhidden))
        #分类器
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.Mask_layer = MaskLayer()


        #取出参数
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = args.dropout
        self.alpha = args.alpha
        self.lam = args.lam
        self.tau = args.tau
        self.lamda = args.lamda

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def bind_loss(self, z1,z2):  # 约束  使用距离约束是有效的说明学出来的特征之间应该是非常相近的
        mask = 1-torch.eye(z1.shape[0]).to(z1.device)
        # 对比学习约束
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))  #
        between_sim = f(self.sim(z1, z2))
        neg_c2 = refl_sim * mask
        neg_c3 = between_sim * mask
        CT = -torch.log( between_sim.diag() / (neg_c2.sum(1) + neg_c3.sum(1)) )
        return CT.mean()

    def forward(self, x, adj):
        _layers = []
        #dropout
        x = F.dropout(x, self.dropout, training=self.training)
        #线性变换 降维
        last_layer = F.relu(self.fcs[0](x))

        _layers.append(last_layer)
        loss = 0
        for i,con in enumerate(self.convs):
            #dropout
            last_layer = F.dropout(last_layer, self.dropout, training=self.training)
            #卷积
            current_layer = self.act_fn(con(last_layer,adj,_layers[0],self.lam,self.alpha,i+1))
            current_layer = self.Mask_layer(current_layer,last_layer,self.lamda,i+1)
            last_layer = current_layer

        layer_inner = F.dropout(last_layer, self.dropout, training=self.training)
        loss+= self.bind_loss(last_layer,last_layer)
        #进行分类
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1),loss,0,0
