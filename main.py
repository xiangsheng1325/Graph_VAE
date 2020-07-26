import os
import igraph
import powerlaw
import pandas as pd
import scipy.sparse as sp
import random
import numpy as np
import torch
import time
import warnings
from collections import defaultdict
import torch.optim as optim
from torch.nn.modules.module import Module
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import SpectralEmbedding
import copy
import matplotlib.pyplot as plt
import networkx as nx
import pprint
from scipy.sparse.csgraph import connected_components
from Graph_VAE.models import *
from Graph_VAE.data_utils import *
from Graph_VAE.eval_utils import *
from Graph_VAE.Options import *
warnings.filterwarnings("ignore")


def get_options():
    opt = Options()
    opt = opt.initialize()
    return opt




if __name__ == "__main__":
    opt = get_options()
    ##{ 临时改超参数
    opt.gpu = '0'
    opt.cond_size = 0
    opt.max_epochs = 500
    opt.gamma = 500
    opt.data_dir = "./data/ENZYMES_20-50_res.graphs"
    ## 正式训练时收起 }
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    print('=========== OPTIONS ===========')
    pprint.pprint(vars(opt))
    print(' ======== END OPTIONS ========\n\n')

    train_adj_mats, test_adj_mats, train_attr_vecs, test_attr_vecs = load_data(
        DATA_FILEPATH=opt.data_dir)

    Encoder = GCNEncoder(
        emb_size=8,
        hidden_dim=16,
        layer_num=2,
    ).cuda()
    Decoder = VanillaDecoder().cuda()
    optim_vae = optim.Adam(Encoder.parameters(), lr=opt.lr)

    training_index = list(range(len(train_adj_mats)))
    print("success!")
