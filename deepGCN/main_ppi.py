import argparse
import csv
import numpy as np
import string
from architecture import DeepGCN
from data import load_data
import torch
import  random
from sklearn.metrics import f1_score
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

from util import train, val_and_test

def generate_random_code(length=6):
    # 生成当前时间戳
    timestamp = int(time.time() * 1000)
    # 生成随机字符串
    random_chars = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    # 将时间戳和随机字符串拼接在一起
    random_code = str(timestamp) + random_chars
    return random_code
def save_result(dataset,model,result):
    path = f'result/{dataset}/{model}'
    os.makedirs(path,exist_ok=True)
    filenmame = f'{model}-{generate_random_code(5)}.csv'
    with open(os.path.join(path,filenmame),'w') as fout:
        fout.write(result)


def load_pretrained_optimizer(pretrained_model, optimizer, scheduler, lr, use_ckpt_lr=True):
    if pretrained_model:
        if os.path.isfile(pretrained_model):
            checkpoint = torch.load(pretrained_model)
            if 'optimizer_state_dict' in checkpoint.keys():
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
            if 'scheduler_state_dict' in checkpoint.keys():
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if use_ckpt_lr:
                    try:
                        lr = scheduler.get_lr()[0]
                    except:
                        lr = lr

    return optimizer, scheduler, lr

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='PyTorch implementation of Deep GCN')

# model args
parser.add_argument('--block', default='res', type=str, help='graph backbone block type {res, dense, plain}')
parser.add_argument('--act', default='relu', type=str, help='activation layer {relu, prelu, leakyrelu}')
parser.add_argument('--norm', default='batch', type=str, help='batch or instance normalization')
parser.add_argument('--bias', default=True,  type=bool, help='bias of conv layer True or False')
parser.add_argument('--n_filters', default=256, type=int, help='number of channels of deep features')
parser.add_argument('--n_blocks', default=14, type=int, help='number of basic blocks')
# convolution
parser.add_argument('--conv', default='gin', type=str, help='graph conv layer {edge, mr, gin, gat, gcn}')
parser.add_argument('--n_heads', default=8, type=int, help='number of heads of GAT')

# saving
parser.add_argument('--data', type=str, default='cora', help='{cora, pubmed, citeseer}.')
parser.add_argument('--model', type=str, default='deepGCN', )
parser.add_argument('--hid', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate.')
parser.add_argument('--epochs', type=int, default=60, help='Number of epochs to train.')
parser.add_argument('--wightDecay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nlayer', type=int, default=8, help='Number of layers, works for Deep model.')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda or dont use cuda')
parser.add_argument("--seed", type=int, default=30, help="seed for model")  # cora 30 citeseer 5 pubmed 0
opt = parser.parse_args()
if opt.cuda==False:
    opt.device = torch.device('cpu')
else:
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data =  load_data(opt.data) #加载数据
nfeat = data.num_features
nclass = int(data.y.max())+1
opt.n_classes = nclass
opt.in_channels = data.x.shape[1]

best_acc = 0
all_test_acc = []

for i in range(1):
    test_acc_list=[]
    set_seed(opt.seed)
    net = DeepGCN(opt).to(opt.device) #加载模型
    net = net.cuda() if opt.cuda==True else net.cpu()


    optimizer = torch.optim.Adam(net.parameters(), opt.lr, weight_decay=opt.wightDecay)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc_value=0
    for epoch in range(opt.epochs):
        train_loss, train_acc= train(net, opt.model, optimizer, criterion, data)
        test_acc = val_and_test(net, opt.model, data)
        test_acc_list.append( round(test_acc.tolist(),3) )
        if test_acc>best_acc_value:
            best_acc_value = test_acc
        print("epoch:{} train_acc:{},train_loss:{},test_acc:{}".format(epoch, round(train_acc.tolist(), 3),
                                                                       round(train_loss.tolist(), 3),
                                                                       round(test_acc.tolist(), 3)))

    all_test_acc.append(best_acc_value)
    rslt_head = f'ACC,Model,nLayers,lr,dropout,parameters\n'
    rslt_values = f'{best_acc_value},{opt.model},' \
                  f'{opt.nlayer},{opt.lr},{opt.dropout},' \
                  f'{opt.__dict__}'
    save_result(opt.data, opt.model, rslt_head + rslt_values)


print("all acc:",all_test_acc)