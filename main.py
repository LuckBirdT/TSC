import argparse
import models
from utils import train, val_and_test
from data import load_data
from metrics import *
import torch
import  random
import os


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



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
parser.add_argument('--lamda', type=float, default=0.5, help='')
parser.add_argument('--tau', type=float, default=0.4, help='')
parser.add_argument('--k', type=int, default=1, help='skip' )
parser.add_argument('--dropout_c', type=float, default=0.2, help='')




args = parser.parse_args()
args.cuda =args.cuda and torch.cuda.is_available()


data =   load_data(args.data)# load_data_pcc(args.data)
nfeat = data.num_features
nclass = int(data.y.max())+1


best_acc = 0
all_test_acc = []

for i in range(1):
    test_acc_list=[]
    set_seed(args.seed)
    net = getattr(models, args.model)(args, nfeat, nclass)
    net = net.cuda() if args.cuda==True else net.cpu()


    optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=args.wightDecay)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc_value=0
    for epoch in range(args.epochs):
        train_loss, train_acc ,_ ,_= train(net, args.model,optimizer, criterion, data)
        test_acc,_,_ = val_and_test(net, args.model,data)
        test_acc_list.append( round(test_acc.tolist(),3) )
        if test_acc>best_acc_value:
            best_acc_value = test_acc
        print("epoch:{} train_acc:{},train_loss:{},test_acc:{}".format(epoch, round(train_acc.tolist(), 3),
                                                                       round(train_loss.tolist(), 3),
                                                                       round(test_acc.tolist(), 3)))

    all_test_acc.append(best_acc_value)


print("all acc:",all_test_acc)