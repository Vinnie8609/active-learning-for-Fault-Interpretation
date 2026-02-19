from .strategy import Strategy
from common_tools import create_logger

class EntropySampling(Strategy):
    def __init__(self, dataset, net):
        super(EntropySampling, self).__init__(dataset, net)

    def query(self, n, seed, otherchoice, flag):
        _, unlabeled_data = self.dataset.get_unlabeled_data()
        data, flag_update = self.predict_prob_EntropySampling(unlabeled_data, n, seed, otherchoice, flag)
        logger = create_logger("/home/user/data/liuyue/active_learning_data/{}_{}/{}/log".format(seed,otherchoice,"EntropySampling"),"pick_idx_{}".format(n))
        logger.info(data)
        return flag_update
