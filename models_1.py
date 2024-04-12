from layers import *
import torch.nn as nn
from  metrics import *
import torch.nn.functional as F

############# our model
class TSC_SGC_P(nn.Module):   #pubmed
    def __init__(self,args, nfeat,  nclass):
        super(TSC_SGC_P, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(args.nlayer):  
            self.convs.append(Part_GraphConvolution())

        self.fcs1=nn.Linear(args.hid, nclass)
        self.fcs0 = nn.Linear(args.hid, args.hid)
        self.encoder = nn.Linear(nfeat, args.hid)
        self.params0 = self.fcs0.parameters()
        self.params1 = self.encoder.parameters()
        self.params2 = self.fcs1.parameters()
        self.lamda = args.lamda
        self.tau = args.tau
        self.dropout_c=args.dropout_c
        if args.k ==1:
            self.k = [args.k * i  for i in range(args.nlayer)]
        else:
            self.k=[]
            for i in range(args.nlayer):
                if args.k**i<args.nlayer:
                    self.k.append(args.k**i)
                else:
                    break
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def bind_loss(self, z1,z2): 
        mask = 1-torch.eye(z1.shape[0]).to(z1.device)
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))  #
        between_sim = f(self.sim(z1, z2))
        neg_c2 = refl_sim * mask
        neg_c3 = between_sim * mask
        CT = -torch.log( between_sim.diag() / (neg_c2.sum(1) + neg_c3.sum(1)) )
        return CT.mean()

    def forward(self, x,adj):
        x=torch.squeeze(x,0)
        loss = 0
        zero = self.encoder(x)
        zero_ = self.fcs0(zero)
        layer_inner = zero_
        for i, con in enumerate(self.convs):
            current = con(layer_inner, adj, self.lamda,i+1)
            if i in self.k:
              loss += self.bind_loss(F.dropout(current,self.dropout_c),F.dropout(layer_inner,self.dropout_c))
            layer_inner = current

        finnal = self.fcs1(layer_inner)
        return F.log_softmax(finnal, dim=1),loss,0,0

class TSC_SGC_C(nn.Module):  # cora、citeseer
    def __init__(self,args, nfeat,  nclass):
        super(TSC_SGC_C, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(args.nlayer):  
            self.convs.append(Part_GraphConvolution())

        self.fcs1=nn.Linear(args.hid, nclass)
        self.encoder = nn.Linear(nfeat, args.hid)
        self.params1 = self.encoder.parameters()
        self.params2 = self.fcs1.parameters()
        self.lamda = args.lamda
        self.tau = args.tau
        self.dropout_c=args.dropout_c
        if args.k ==1:
            self.k = [args.k * i  for i in range(args.nlayer)]
        else:
            self.k=[]
            for i in range(args.nlayer):
                if args.k**i<args.nlayer:
                    self.k.append(args.k**i)
                else:
                    break
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def bind_loss(self, z1,z2):  
        mask = 1-torch.eye(z1.shape[0]).to(z1.device)
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))  #
        between_sim = f(self.sim(z1, z2))
        neg_c2 = refl_sim * mask
        neg_c3 = between_sim * mask
        CT = -torch.log( between_sim.diag() / (neg_c2.sum(1) + neg_c3.sum(1)) )
        return CT.mean()

    def forward(self, x,adj):
        x=torch.squeeze(x,0)
        loss = 0
        zero = self.encoder(x)
        layer_inner = zero
        for i, con in enumerate(self.convs):
            current = con(layer_inner, adj, self.lamda, i + 1)
            if i in self.k:
              loss += self.bind_loss(F.dropout(current,self.dropout_c),F.dropout(layer_inner,self.dropout_c))
            layer_inner = current

        finnal = self.fcs1(layer_inner)
        return F.log_softmax(finnal, dim=1),loss,0,0


class TSC_GCN(nn.Module):  #GCN + Random Mask+CC
    def __init__(self, args, nfeat, nclass, **kwargs):
        super(TSC_GCN, self).__init__()
        assert args.nlayer >= 1
        self.hidden_layers = nn.ModuleList([
                GCNLayer(nfeat if i == 0 else args.hid,args.hid)
                for i in range(args.nlayer-1)
            ])
        self.out_layer =  GCNLayer(nfeat if args.nlayer == 1 else args.hid, nclass)

        self.dropout_rate = args.dropout
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.tau= args.tau
        self.lamda=args.lamda
        self.dropout_c = args.dropout_c

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def bind_loss(self, z1,z2): 
        mask = 1-torch.eye(z1.shape[0]).to(z1.device)
        f = lambda x: torch.exp(x / self.tau)
        refl_sim1 = f(self.sim(z1, z1))
        refl_sim2 = f(self.sim(z2, z2))
        between_sim = f(self.sim(z1, z2))
        neg_c1 = refl_sim1 * mask
        neg_c2 = refl_sim2* mask
        CT = -torch.log( between_sim.diag()/ (neg_c1.sum(1) + neg_c2.sum(1)) )
        return CT.mean()

    def get_mask(self, x, drop_prob):
        drop_mask = torch.empty(
            (x.size(1),),
            dtype=torch.float32,
            device=x.device).uniform_(0, 1) < drop_prob
        adding = torch.zeros_like(drop_mask)
        adding[-1] = True
        random.shuffle(adding)
        return drop_mask + adding

    def random_mask(self, last, current,rate,l):
        rate = math.log(rate/(l)+1)
        mask = self.get_mask(current, rate).long()
        x = (current)*(mask)+last*(1-mask)
        return x

    def forward(self, x, adj):
        loss=0
        res_rate=math.log(self.lamda/1+1)
        for l, layer in enumerate(self.hidden_layers):
            if l >= 1:
                x = self.dropout(x)
                current= layer(x, adj)
                x=self.random_mask(x,current,self.lamda,l+1)
                x = mask * x + (1 - mask) * one
                if l ==len(self.hidden_layers)-1:
                    loss += self.bind_loss(F.dropout(x, self.dropout_c), F.dropout(x, self.dropout_c))
            else:
                x = self.dropout(x)
                x = layer(x, adj)
                one = x
                mask = self.get_mask(x, res_rate).long()

        x = F.relu(x)
        x = self.dropout(x)
        x = self.out_layer(x, adj)
        return x,loss,0,0
