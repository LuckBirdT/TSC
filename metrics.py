import torch
from einops import rearrange
import pdb
import numpy as np
from scipy.stats import gamma
import torch.nn.functional as F


def calculate_MAD(x):  #平均距离测量
    x_norm=F.normalize(x)
    mad = 1-torch.mm(x_norm,x_norm.t())
    return mad.mean()
#
# def calculate_MAD(x):  #平均距离测量
#     x_norm=F.normalize(x)
#     mad = torch.cdist(x_norm,x_norm)
#     return mad.mean()

def calculate_MAD_dist(x):  #平均距离测量
    x_norm=F.normalize(x)
    mad = 1-torch.mm(x_norm,x_norm.t())
    return mad.mean()


def calculate_cos(second,layer_k):  #平均距离测量
    cos = torch.mm(F.normalize(second),F.normalize(layer_k).t()).diag()
    return cos.mean()


def jensen_shannon_distance(p, q, epsilon=1e-12):
        p =  F.sigmoid(p)
        q = F.sigmoid(q)
        # 防止分母为零
        p = p.clamp(epsilon, 1.0 - epsilon)
        q = q.clamp(epsilon, 1.0 - epsilon)
        # 计算平均分布
        m = 0.5 * (p + q)
        # 计算相对熵
        kl_divergence_pm = F.kl_div(p.log(), m, reduction='mean')
        kl_divergence_qm = F.kl_div(q.log(), m, reduction='mean')
        # 詹森-香农距离
        js_distance = 0.5 * (kl_divergence_pm + kl_divergence_qm)
        return js_distance.item()
