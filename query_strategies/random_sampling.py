import numpy as np
from .strategy import Strategy
from common_tools import create_logger 

class RandomSampling(Strategy):
    def __init__(self, dataset, net):
        super(RandomSampling, self).__init__(dataset, net)

    def query(self, n,seed,otherchoice,picknum,picknum_no,flag):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        data,flag_update = self.predict_prob_RandomSampling(unlabeled_data,n,seed,otherchoice,picknum,picknum_no,flag)
        # probs_sorted, idxs = probs.sort(descending=False)
        logger = create_logger("/home/user/data/liuyue/active_learning_data/{}_{}/{}/log".format(seed,otherchoice,"RandomSampling"),"pick_idx_{}".format(n))
        logger.info(data)
        return flag_update
  #np.where：labeled_idxs==0的位置，array([0, 1, 4])    1000个
  #np.random.choice: array([4, 1]),   n=10个