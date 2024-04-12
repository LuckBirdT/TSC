import argparse
import models
from utils import train, val_and_test
from data import load_data
from metrics import *
import torch
import  random
import os
import time,string


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
parser.add_argument('--model', type=str, default='GCN', help='{SGC, GCN, GCNII,CGCN}')
parser.add_argument('--hid', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.6,help='Dropout rate.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--wightDecay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nlayer', type=int, default=2, help='Number of layers, works for Deep model.')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda or dont use cuda')
parser.add_argument("--seed",type=int,default=30,help="seed for model")

# for GCNII  
parser.add_argument('--alpha', type=float, default=0.1, help='GCNII  rate para.')
parser.add_argument('--lam', type=float, default=0.5, help='GCNII  rate para.')
parser.add_argument('--variant', type=bool, default=False, help='GCNII  rate para.')
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
#GAT
parser.add_argument('--head', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--al', type=float, default=0.2, help='Alpha for the leaky_relu.')


# for CGCN
parser.add_argument('--lamda', type=float, default=2, help='CGCN  rate para.')
parser.add_argument('--tau', type=float, default=0.4, help='')
parser.add_argument('--k', type=int, default=1, help='skip' )



#读取参数
args = parser.parse_args()
# 固定种子
#set_seed(args.seed)
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
all_best_mad = []
all_best_bais=[]
all_best_shang=[]
arr = [2,4,8,16,32]
for i in range(1):

    test_acc_list=[]
    # args.nlayer =arr[i]
    set_seed(args.seed)
    #用于返回模型对象属性  传入参数 超参数、结点维数、类别数
    net = getattr(models, args.model)(args, nfeat, nclass)
    net = net.cuda() if args.cuda==True else net.cpu()

    # 优化器
    if args.model=="GCNII" or args.model ==  "GCNII_TSC":
        optimizer =  torch.optim.Adam([
            {'params': net.params1, 'weight_decay': args.wd1},
            {'params': net.params2, 'weight_decay': args.wd2},
        ], lr=args.lr)
        criterion = F.nll_loss
    else:
        optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=args.wightDecay)
        criterion = torch.nn.CrossEntropyLoss()

    best_acc_value=0
    best_mad=0

    for epoch in range(args.epochs):
        #训练
        train_loss, train_acc ,_ ,_= train(net, args.model,optimizer, criterion, data)
        #验证和测试
        test_acc,mad,feature_k = val_and_test(net, args.model,data)
        #记录测试结果
        test_acc_list.append( round(test_acc.tolist(),3) )

        if test_acc>best_acc_value:
            best_acc_value = test_acc
        #打印结果
        print("epoch:{} train_acc:{},train_loss:{},test_acc:{}".format(epoch,round(train_acc.tolist(),3),round(train_loss.tolist(),3),round(test_acc.tolist(),3)))
        # 保存验证集最好准确率或损失下的模型参数
    # all_best_mad.append(best_mad)
    all_test_acc.append(best_acc_value)
    print("\ncurrent max test acc",best_acc_value)
    rslt_head=f'ACC,MAD,Model,nLayers,lr,dropout,parameters\n'
    rslt_values = f'{best_acc_value},{best_mad},{args.model},' \
           f'{args.nlayer},{args.lr},{args.dropout},' \
           f'{args.__dict__}'
    save_result(args.data,args.model,rslt_head+rslt_values)

print(all_test_acc)