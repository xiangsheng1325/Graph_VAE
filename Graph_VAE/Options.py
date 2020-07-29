# Options
class SimpleOpt():
    def __init__(self):
        self.method = 'pool_link'
        self.dataset = 'dblp'
        self.data_dir = './data/dblp/'
        self.cond_size = 10
        self.emb_size = 2
        self.hidden_dim = 10
        self.output_dim = 34
        self.n1 = 80
        self.n2 = 40
        self.rep_size = 32
        self.gcn_output_size = 16
        # self.adj_thresh = 0.6
        self.max_epochs = 10
        self.lr = 0.003
        self.beta = 5
        self.beta2 = 0.1
        self.alpha = 0.1
        self.gamma = 15
        self.gpu = '2'
        self.mbs = 16
        self.batch_size = 56
        self.graph_type = 'ENZYMES'
        self.epochs_log = 1
        # self.DATA_DIR = './data/dblp/'
        # self.output_dir = './output/'


class Options():
    def __init__(self):
        self.opt_type = 'simple'

    def initialize(self, epoch_num=1, cond_size=0):
        opt = SimpleOpt()
        opt.max_epochs = epoch_num
        opt.cond_size = cond_size
        return opt

