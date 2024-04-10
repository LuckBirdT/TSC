import torch
from einops import rearrange
import pdb
import numpy as np
from scipy.stats import gamma
import torch.nn.functional as F


def calculate_MAD(x): 
    x_norm=F.normalize(x)
    mad = 1-torch.mm(x_norm,x_norm.t())
    return mad.mean()


def calculate_MAD_dist(x):  
    x_norm=F.normalize(x)
    mad = 1-torch.mm(x_norm,x_norm.t())
    return mad.mean()


def calculate_cos(second,layer_k):  
    cos = torch.mm(F.normalize(second),F.normalize(layer_k).t()).diag()
    return cos.mean()


