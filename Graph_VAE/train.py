import random


def train_vae_epoch(epoch, args, model, dataloader, optimizer):
    loss = 0
    graph_index = -1
    for graph_index, adj in enumerate(zip((dataloader['adj_mats']))):
        rec_adj = model(adj)
    print("Epoch:{}/{}, ".format(epoch, args.max_epoch))
    return loss/(graph_index+1)