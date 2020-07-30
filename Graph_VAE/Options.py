# Options
class SimpleOpt():
    def __init__(self):
        self.method = 'gvae'
        self.graph_type = 'ENZYMES'
        self.data_dir = './data/ENZYMES_20-50_res.graphs'
        self.emb_size = 8
        self.encode_dim = 32
        self.layer_num = 3
        self.decode_dim = 32
        self.dropout = 0.5
        self.logits = 10
        # self.adj_thresh = 0.6
        self.max_epochs = 50
        self.lr = 0.003
        self.gpu = '2'
        self.batch_size = 56
        self.epochs_log = 1
        # self.DATA_DIR = './data/dblp/'
        # self.output_dir = './output/'


class Options():
    def __init__(self):
        self.opt_type = 'simple'
        # self.opt_type = 'argparser

    @staticmethod
    def initialize(epoch_num=1800):
        opt = SimpleOpt()
        opt.max_epochs = epoch_num
        return opt

