# models
# GCN

class GCN(Module):
    """
    GCN layer, reference to: https://arxiv.org/abs/1609.02907
    """

    def __init__(self, input_len, output_len, bias=True):
        super(GCN, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        # print("Type of ouput_len: {}\t, value of output_len: {}".format(type(output_len), output_len))
        self.W = Parameter(torch.FloatTensor(input_len, output_len))
        if bias:
            self.b = Parameter(torch.FloatTensor(output_len))
        else:
            self.register_parameter('b', None)

    def reset_parameters(self):
        """
        reset parameters in uniform distribution
        """
        margin = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-margin, margin)
        if self.b is not None:
            self.b.data.uniform_(-margin.margin)

    def forward(self, inputs, adj):
        """
        When this layer is called, execute this function.
        :param inputs:  embedded node vectors (with condition context)
        :param adj:  adjacent matrix
        :return: output of GCN layer
        """
        support = torch.mm(inputs, self.W)
        # output = torch._sparse_mm(adj, support)
        output = torch.spmm(adj, support)
        if self.b is None:
            return output
        else:
            return output + self.b

    def __str__(self):
        return "Layer: {}({}->{})".format(self.__class__.__name__, self.input_len, self.output_len)

    def __repr__(self):
        return "Layer: {}({}->{})".format(self.__class__.__name__, self.input_len, self.output_len)


class GCNEncoder(nn.Module):
    def __init__(self, emb_size, hidden_dim, layer_num=2):
        super(GCNEncoder, self).__init__()
        self.act = nn.ReLU()
        self.gc_top = GCN(emb_size, hidden_dim)
        self.gc_medium = [GCN(hidden_dim, hidden_dim) for i in range(layer_num-2)]
        self.gc_bottom = GCN(hidden_dim, hidden_dim)

    def forward(self, x, adj, normalized=True):
        """
        call this function when you use encoder
        :param x: node features of one graph
        :param adj: Normalized adjacency matrix of one graph.
        :return:
        """
        if not normalized:
            adj = normalize(adj)
        x = self.gc_top(x, adj)
        x = self.act(x)
        for gcn in self.gc_medium:
            x = gcn(x, adj)
            x = self.act(x)
        x = self.gc_bottom(x, adj)
        return self.act(x)


class VanillaDecoder(nn.Module):
    def __init__(self, dropout=0.5):
        """
        InnerProductDecoder definition.
        :param dropout: probability to randomly zero some of the elements from a Bernoulli distribution.
        """
        super(VanillaDecoder, self).__init__()
        self.act = nn.Sigmoid()
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, p=self.dropout)
        rec = torch.mm(x, x.permute(1, 0))
        return self.act(rec)


class MLPDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, dropout=0.5):
        super(MLPDecoder, self).__init__()
        self.act = nn.Sigmoid()
        self.dropout = dropout
        self.decode = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.BatchNorm1d(hidden_dim),
                                    nn.ReLU())

    def forward(self, x):
        x = F.dropout(x, p=self.dropout)
        x = self.decode(x)
        rec = torch.mm(x, x.permute(1, 0))
        return self.act(x)
