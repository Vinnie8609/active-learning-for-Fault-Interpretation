import numpy as np
from .strategy import Strategy
from common_tools import create_logger 

class MarginSampling(Strategy):
    def __init__(self, dataset, net):
        super(MarginSampling, self).__init__(dataset, net)

    def query(self, n,seed,otherchoice,picknum,picknum_no,flag):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        data,flag_update = self.predict_prob_MarginSampling(unlabeled_data,n,seed,otherchoice,picknum,picknum_no,flag)
        # probs_sorted, idxs = probs.sort(descending=False)
        logger = create_logger("/home/user/data/liuyue/active_learning_data/{}_{}/{}/log".format(seed,otherchoice,"MarginSampling"),"pick_idx_{}".format(n))
        logger.info(data)
        # id=unlabeled_idxs[idxs[:n]]
        # logger.info(id)
        # return id
        return flag_update

