import random

import torch, os
import numpy as np 
import scipy.sparse as sp
import pickle as pkl
import networkx as nx
import sys
from torch_geometric.data import Data
from dgl.data import CoauthorCSDataset,AmazonCoBuyPhotoDataset,AmazonCoBuyComputerDataset,CoauthorPhysicsDataset
from dgl.data import CornellDataset,WisconsinDataset
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# 加载数据  通过API  torch_geometric.datasets  导入数据集
def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    if dataset_str in ['pubmed', 'citeseer', 'cora'] :
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/{}/ind.{}.{}".format(dataset_str,dataset_str,names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/{}/ind.{}.test.index".format(dataset_str,dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        test_mask = test_idx_range.tolist()
        train_mask = range(len(y))
        val_mask = range(len(y), len(y)+500)
        #创建图数据
        data = Data()
        data.x = torch.FloatTensor(normalize_features(features))
        data.y = torch.argmax(torch.FloatTensor(labels),dim=1)
        data.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj( adj+ sp.eye(adj.shape[0]) ))
        data.adj_DA = sparse_mx_to_torch_sparse_tensor(normalize_adj_row( adj+ sp.eye(adj.shape[0]) ))
        data.src_trg_edges = data.adj._indices()
        data.test_mask = test_mask
        data.train_mask = train_mask
        data.val_mask = val_mask
    elif dataset_str in ['CoauthorCS', 'AmazonPhoto','CoauthorPhysics','AmazonComputer']:
        if dataset_str == 'CoauthorCS':
            dataset = CoauthorCSDataset('./data')
        elif dataset_str == 'AmazonPhoto':
            dataset = AmazonCoBuyPhotoDataset('./data')
        elif dataset_str == "CoauthorPhysics":
            dataset = CoauthorPhysicsDataset('./data')
        elif dataset_str == 'AmazonComputer':
            dataset = AmazonCoBuyComputerDataset('./data')
        data = dataset[0]
        num_class = dataset.num_classes
        feat = data.ndata['feat']  # get node feature
        label = data.ndata['label']# get node labels
        adj = data.adjacency_matrix().to_dense()+ torch.eye(feat.shape[0])
        adj =  sparse_mx_to_torch_sparse_tensor(normalize_adj(adj)) #获得邻接矩阵

        test_mask = range(feat.shape[0]-1000,feat.shape[0])
        train_mask = get_per_class_node(range(0,int(feat.shape[0]/2)),label,num_class,20)
        val_mask = range(int(feat.shape[0]/2),int(feat.shape[0]/2)+500)
        data = Data()
        data.x = feat
        data.adj = adj
        data.y = label
        data.train_mask = train_mask
        data.val_mask = list(val_mask)
        data.test_mask = list(test_mask)
    elif dataset_str in ['Wisconsin', 'Cornell']:
        if dataset_str=='Cornell':
            dataset = CornellDataset('./data')
        elif dataset_str=='Wisconsin':
            dataset = WisconsinDataset('./data')
        da = dataset[0]
        num_class = dataset.num_classes
        feat = da.ndata['feat']  # get node feature
        label = da.ndata['label']  # get node labels
        adj = da.adjacency_matrix().to_dense() + torch.eye(feat.shape[0])
        adj = sparse_mx_to_torch_sparse_tensor(normalize_adj(adj))  # 获得邻接矩阵
        data = Data()
        data.x = feat
        data.adj = adj
        data.y = label
        train_mask = get_per_class_node(range(0, int(feat.shape[0]-50)), label, num_class, 5)
        test_mask =  list( range(feat.shape[0]-50,feat.shape[0]) )
        data.train_mask=train_mask
        data.test_mask=test_mask


    return  data.cuda() if torch.cuda.is_available() else data

def get_per_class_node(range_node,label,num_class,num_per=20):
    arr= [0]*num_class
    # min =label.min()
    # max = label.max()
    mask=[]
    for i in range_node:
       if  arr[label[i]] <num_per  and sum(arr)<num_per*num_class:
           mask.append(i)
           arr[label[i]] += 1
    return mask


#对称归一化邻接矩阵 D^(-0.5)AD^(-0.5)
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

#非对称归一化  D^(-1)A
def normalize_adj_row(adj):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(adj)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


#遗失结点特征
def missingNodesFeature(data,missing_rate,Data_root,data_name):
    # generate missing feature setting   生成缺失特征
    n=len(data.x)
    indices_dir = os.path.join(Data_root, data_name, 'indices')  # 目录拼接
    if not os.path.isdir(indices_dir):  # 检测是否为目录
        os.mkdir(indices_dir)  # 创建目录
    missing_indices_file = os.path.join(indices_dir, "indices_missing_rate={}.npy".format(missing_rate))  # 创建文件路径
    if not os.path.exists(missing_indices_file):  # 如果文件不存在,则创建missing_rate的结点遗失文件
        temp = ~data.train_mask  # ~取反运算符， 相当于把非训练数据筛选出来
        erasing_pool = torch.arange(n)[temp]  # keep training set always full feature  #torch.arange(n) 生成tensor([0,1,2,3,4...n-1])
        size = int(len(erasing_pool) * (missing_rate / 100))
        # 随机1选择size个非训练结点
        idx_erased = np.random.choice(erasing_pool, size=size, replace=False)
        np.save(missing_indices_file, idx_erased)  # 把要遗失的结点号装入到该文件里
    else:  # 如果文件存在，直接加载
        idx_erased = np.load(missing_indices_file)  # idx_reased 里面装的是要遗失的结点
    # 擦除特征
    if missing_rate > 0:  # 如果遗失率大于0 则遗失那些要遗失的结点
        data.x[idx_erased] = 0  # 将要遗失的结点的特征每个维度上的值变为0

    return data  #其实可以不用返回值 为了统一操作设置一个返回值

#特征归一化 单位化
def normalize_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()
