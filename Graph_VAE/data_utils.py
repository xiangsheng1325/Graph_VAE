def get_spectral_emb(adj, max_size):
    """
    Given adj is N*N, return its feature mat N*D, D is fixed in model
    :param adj: adjacent matrix
    :param max_size: the amount of dimension to be embedded
    :return: spectral embedding of every node in this graph
    """

    adj_ = adj.data.cpu().numpy()
    emb = SpectralEmbedding(n_components=max_size)
    res = emb.fit_transform(adj_)
    x = torch.from_numpy(res).float().cuda()
    return x


def get_embedding(adj, max_size=2, method='spectral'):
    if method == 'spectral':
        return get_spectral_emb(adj, max_size)
    return adj[:, :max_size]


def normalize(adj):
    adj = adj.data.cpu().numpy()
    adj_ = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    degree_mat_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = degree_mat_inv_sqrt.dot(adj_).dot(degree_mat_sqrt)
    return torch.from_numpy(adj_normalized).float().cuda()


def cat_cond(x, cond_vec):
    if cond_vec is None:
        return x
    attr_mat = cond_vec.repeat(x.size()[0], 1)
    x = torch.cat([x, attr_mat], dim=1)
    return x


def show_graph(adj, base_adj=None, remove_isolated=True, show=False):
    if not isinstance(adj, np.ndarray):
        adj_ = adj.data.cpu().numpy()
    else:
        adj_ = copy.deepcopy(adj)

    adj_ -= np.diag(np.diag(adj_))

    assert ((adj_ == adj_.T).all())
    if show:
        gr = nx.from_numpy_array(adj_)
        if remove_isolated:
            gr.remove_nodes_from(list(nx.isolates(gr)))
        nx.draw(gr, node_size=10)
        plt.title('gen')
        plt.show()

    d = compute_graph_statistics(adj_)
    #pprint.pprint(d)

    if base_adj is not None:
        base_gr = nx.from_numpy_array(base_adj)
        if show:
            nx.draw(base_gr, node_size=10)
            plt.title('base')
            plt.show()
        bd = compute_graph_statistics(base_adj)
        diff_d = {}
        for k in list(d.keys()):
            diff_d[k] = round(abs(d[k] - bd[k]), 4)
        # print(diff_d.keys())
        # print(diff_d.values())
        return diff_d


def keep_topk_conns(adj, k=3):
    g = nx.from_numpy_array(adj)
    to_removes = [cp for cp in sorted(nx.connected_components(g), key=len)][:-k]
    for cp in to_removes:
        g.remove_nodes_from(cp)
    adj = nx.to_numpy_array(g)
    return adj


def top_n_indexes(arr, n):
    idx = np.argpartition(arr, arr.size - n, axis=None)[-n:]
    width = arr.shape[1]
    return [divmod(i, width) for i in idx]


def topk_adj(adj, k):
    adj_ = adj.data.cpu().numpy()
    #print("adj prob:\n{}".format(adj_))
    assert ((adj_ == adj_.T).all())
    adj_ = (adj_ - np.min(adj_)) / np.ptp(adj_)
    adj_ -= np.diag(np.diag(adj_))
    # print("prob:\n{}".format(adj))
    tri_adj = np.triu(adj_)
    inds = top_n_indexes(tri_adj, k // 2)
    res = torch.zeros(adj.shape)
    for ind in inds:
        i = ind[0]
        j = ind[1]
        res[i, j] = 1.0
        res[j, i] = 1.0
    #print("adj:\n{}".format(res))
    return res.cuda()


def gen_graph(model, n, e, attr_vec, z_size, opt):
    fixed_noise = torch.randn((n, z_size), requires_grad=True).cuda()
    # print("the shape of attr_gen1: {}".format(fixed_noise.data.cpu().numpy().shape))
    if opt.cond_size:
        fixed_noise = cat_cond(fixed_noise, attr_vec)
    # print("the shape of attr_gen2: {}".format(fixed_noise.data.cpu().numpy().shape))
    rec_adj = model.decoder(fixed_noise)
    return topk_adj(F.sigmoid(rec_adj), e * 2)
