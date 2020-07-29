import time
import warnings
import pprint
from Graph_VAE.data_utils import *
from Graph_VAE.train import *
from Graph_VAE.Options import *
warnings.filterwarnings("ignore")


def get_options():
    opt = Options()
    opt = opt.initialize()
    return opt


def timelog(func):
    print("This is a time logger.")

    def printtime(*args, **argv):
        t1 = time.time()
        print("Start time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(t1))))
        returns = func(*args, **argv)
        t2 = time.time()
        print("End time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(t2))))
        print("Time consumption: {}s".format(t2-t1))
        return returns
    return printtime


if __name__ == "__main__":
    opt = get_options()
    ##{ 临时改超参数
    opt.gpu = '0'
    opt.cond_size = 0
    opt.max_epochs = 1800
    opt.gamma = 500
    opt.data_dir = "./data/ENZYMES_20-50_res.graphs"
    ## 正式训练时收起 }
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    print('=========== OPTIONS ===========')
    pprint.pprint(vars(opt))
    print(' ======== END OPTIONS ========\n\n')

    train_adj_mats, test_adj_mats, train_attr_vecs, test_attr_vecs = load_data(
        DATA_FILEPATH=opt.data_dir)
    with torch.autograd.set_detect_anomaly(True):
        train(
            opt=opt,
            train_adj_mats=train_adj_mats
        )
    # todo: write testing process after all training process.
    print("success!")
