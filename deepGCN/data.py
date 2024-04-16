import torch, os
import numpy as np 
import scipy.sparse as sp
import pickle as pkl
import networkx as nx
import sys
from torch_geometric.data import Data
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

#
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
        data = Data()
        data.x = torch.FloatTensor(normalize_features(features))
        data.y = torch.argmax(torch.FloatTensor(labels),dim=1)
        data.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj( adj+ sp.eye(adj.shape[0]) ))
        # data.adj_DA = sparse_mx_to_torch_sparse_tensor(normalize_adj_row( adj+ sp.eye(adj.shape[0]) ))
        data.src_trg_edges = data.adj._indices()
        data.test_mask = test_mask
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.edge_index = data.src_trg_edges
    return  data.cuda() if torch.cuda.is_available() else data


#D^(-0.5)AD^(-0.5)
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

# D^(-1)A
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

def normalize_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()
