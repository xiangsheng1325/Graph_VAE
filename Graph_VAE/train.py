import random
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from Graph_VAE.models import *
import torch.optim as optim


def train_vae_epoch(epoch, args, model, train_adj_mats, optimizer, scheduler):
    loss = 0
    loss_kl_sum = 0
    loss_bce_sum = 0
    loss_sum = 0
    graph_index = -1
    batch_index = 0
    confus_all = np.array([[0, 0], [0, 0]])
    adj_indexs = list(range(len(train_adj_mats)))
    random.shuffle(adj_indexs)
    for graph_index, adj_index in enumerate(adj_indexs):
        adj = Variable(torch.from_numpy(train_adj_mats[adj_index]).float()).cuda()
        batch_index = batch_index + 1
        loss_kl, loss_bce, confus_m = model(adj, normalized=False, training=True)
        # print("Shape of kl_loss: {}, value: {}".format(loss_kl.shape, loss_kl.data))
        # print("Shape of bce_loss: {}, value: {}".format(loss_bce.shape, loss_bce.data))
        loss = loss + loss_kl + loss_bce
        loss_kl_sum = loss_kl_sum + loss_kl
        loss_bce_sum = loss_bce_sum + loss_bce
        # print(confus_all, "\n", confus_m)
        confus_all = confus_all + confus_m
        loss_sum = loss_sum + loss
        if batch_index == args.batch_size:
            loss /= batch_index
            batch_index = 0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = 0
    loss_sum /= (graph_index + 1)
    scheduler.step()
    if epoch % args.epochs_log==0:
        loss_kl_sum /= (graph_index + 1)
        loss_bce_sum /= (graph_index + 1)
        confus_all = confus_all.ravel().tolist()
        all_instances = sum(confus_all)
        true_negative, false_positive, false_negative, true_positive = confus_all
        print("Epoch: {}/{}, bce loss: {:.5f}, kl loss: {:.5f}, graph_type: {}, acc: {:.4f}, prc: {:.4f}, rec: {:.4f}".format(
            epoch,
            args.max_epochs,
            loss_bce_sum.data,
            loss_kl_sum.data,
            args.graph_type,
            (true_negative+true_positive)/all_instances,
            true_positive/(true_positive+false_positive),
            true_positive/(true_positive+false_negative),
        ))
    # return whatever you like.
    return loss_sum


def train(opt, train_adj_mats):

    model = GraphVAE(
        emb_size=8,
        encode_dim=16,
        layer_num=2,
        decode_dim=16,
        dropout=0.5,
    ).cuda()
    optim_vae = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler_vae = MultiStepLR(optim_vae, milestones=[300, 700, 1200], gamma=0.3)

    for epoch in range(1, opt.max_epochs+1):
        train_vae_epoch(
            epoch=epoch,
            args=opt,
            model=model,
            train_adj_mats=train_adj_mats,
            optimizer=optim_vae,
            scheduler=scheduler_vae,
        )
    return model
