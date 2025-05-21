import numpy as np
from .strategy import Strategy
from common_tools import create_logger 

class LeastConfidence(Strategy):
    def __init__(self, dataset, net):
        super(LeastConfidence, self).__init__(dataset, net)

    def query(self, n,seed,otherchoice,picknum,picknum_no,flag):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        data,flag_update = self.predict_prob_LeastConfidence(unlabeled_data,n,seed,otherchoice,picknum,picknum_no,flag)
        # probs_sorted, idxs = probs.sort(descending=True)
        # logger = create_logger("/home/user/data/liuyue/deep-active-learning-master/deep-active-learning-master/log/","LeastConfidence_idx")
        # id=unlabeled_idxs[idxs[:n]]
        # logger.info(id)
        # return id
        logger = create_logger("/home/user/data/liuyue/active_learning_data/{}_{}/{}/log".format(seed,otherchoice,"LeastConfidence"),"pick_idx_{}".format(n))
        logger.info(data)
        return flag_update
        
