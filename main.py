import argparse
import models
from utils import train, val_and_test
from data import load_data
from metrics import *
import torch
import  random
import os

#设置种子
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 配置参数
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cora', help='{cora, pubmed, citeseer}.')
parser.add_argument('--model', type=str, default='TSC_GCN', help='{TSC_SGC_C,TSC_SGC_P,TSC_GCN}')
parser.add_argument('--hid', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.6,help='Dropout rate.')
parser.add_argument('--epochs', type=int, default=60, help='Number of epochs to train.')
parser.add_argument('--wightDecay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nlayer', type=int, default=8, help='Number of layers, works for Deep model.')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda or dont use cuda')
parser.add_argument("--seed",type=int,default=30,help="seed for model")  # cora 30 citeseer 5 pubmed 0

# for TSC
parser.add_argument('--lamda_c', type=float, default=0.5, help='')
parser.add_argument('--lr1', type=float, default=0.01, help='encoder')
parser.add_argument('--lr2', type=float, default=0.01, help='class')
parser.add_argument('--tau', type=float, default=0.4, help='')
parser.add_argument('--k', type=int, default=1, help='skip' )
parser.add_argument('--dropout_c', type=float, default=0.2, help='')



#读取参数
args = parser.parse_args()
#检测cuda是否可用并设置cuda的值
args.cuda =args.cuda and torch.cuda.is_available()


#加载数据 返回： 单位化X   归一化adj  训练集验证集测试集mask
data =   load_data(args.data)# load_data_pcc(args.data)
#获得结点向量的维数
nfeat = data.num_features
#获得类别数
nclass = int(data.y.max())+1

#最佳准确率
best_acc = 0
#用于存储测试指标
all_test_acc = []

for i in range(1):
    test_acc_list=[]
    set_seed(args.seed)
    #用于返回模型对象属性  传入参数 超参数、结点维数、类别数
    net = getattr(models, args.model)(args, nfeat, nclass)
    net = net.cuda() if args.cuda==True else net.cpu()

    # 优化器
    if args.model == "TSC_SGC_P" :
        optimizer = torch.optim.Adam([{'params':net.params1,'weight_decay':0.0,'lr':args.lr1},  #编码cora 0.005  cite 0.005
                                      {'params':net.params2,'weight_decay':0.00,'lr':args.lr2},  #分类    cora 0.1 cite 0.1
                                      {'params':net.params0,'weight_decay':0.00,'lr':args.lr1},
                                      ])
        criterion = F.nll_loss
    elif args.model == "TSC_SGC_C" :
        optimizer = torch.optim.Adam(
            [{'params': net.params1, 'weight_decay': 0.0, 'lr': args.lr1},  # 编码cora 0.005  cite 0.005
             {'params': net.params2, 'weight_decay': 0.00, 'lr': args.lr2},  # 分类    cora 0.1 cite 0.1
             ])
        criterion = F.nll_loss
    else:
        optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=args.wightDecay)
        criterion = torch.nn.CrossEntropyLoss()

    best_acc_value=0
    for epoch in range(args.epochs):
        #训练
        train_loss, train_acc ,_ ,_= train(net, args.model,optimizer, criterion, data)
        test_acc,_,_ = val_and_test(net, args.model,data)
        #记录测试结果
        test_acc_list.append( round(test_acc.tolist(),3) )
        if test_acc>best_acc_value:
            best_acc_value = test_acc
        print("epoch:{} train_acc:{},train_loss:{},test_acc:{}".format(epoch, round(train_acc.tolist(), 3),
                                                                       round(train_loss.tolist(), 3),
                                                                       round(test_acc.tolist(), 3)))

    all_test_acc.append(best_acc_value)


print("all acc:",all_test_acc)