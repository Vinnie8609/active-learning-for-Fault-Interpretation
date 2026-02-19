class Strategy:
    def __init__(self, dataset, net):
        self.dataset = dataset
        self.net = net

    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    def train_before(self, n, strategy_name, seed, otherchoice):
        labeled_data = self.dataset.get_labeled_data()
        val_data = self.dataset.get_val_data()
        best_iou = self.net.train_before(labeled_data, val_data, n, strategy_name, seed, otherchoice)
        return best_iou

    def train(self, n, strategy_name, best_iou, seed, otherchoice):
        labeled_data = self.dataset.get_labeled_data()
        val_data = self.dataset.get_val_data()
        best_iou = self.net.train(labeled_data, val_data, n, strategy_name, best_iou, seed, otherchoice)
        return best_iou

    def predict(self, data, strategy_name):
        self.net.predict(data, strategy_name)

    def predict_prob_Orderselect(self, data, n, seed, otherchoice, picknum, picknum_no):
        probs = self.net.predict_prob_Orderselect(data, n, seed, otherchoice, picknum, picknum_no)
        return probs

    def predict_prob_RandomSampling(self, data, n, seed, otherchoice, picknum, picknum_no, flag):
        probs = self.net.predict_prob_RandomSampling(data, n, seed, otherchoice, picknum, picknum_no, flag)
        return probs

    def predict_prob_MarginSampling(self, data, n, seed, otherchoice, picknum, picknum_no, flag):
        probs = self.net.predict_prob_MarginSampling(data, n, seed, otherchoice, picknum, picknum_no, flag)
        return probs

    def predict_prob_EntropySampling(self, data, n, seed, otherchoice, flag):
        probs = self.net.predict_prob_EntropySampling(data, n, seed, otherchoice, flag)
        return probs

    def predict_prob_LeastConfidence(self, data, n, seed, otherchoice, picknum, picknum_no, flag):
        probs = self.net.predict_prob_LeastConfidence(data, n, seed, otherchoice, picknum, picknum_no, flag)
        return probs

    def predict_prob_MaxconfidenceSampling(self, data, n, seed, otherchoice, picknum, picknum_no):
        probs = self.net.predict_prob_MaxconfidenceSampling(data, n, seed, otherchoice, picknum, picknum_no)
        return probs

    def predict_prob_MarginSampling_dropout(self, data, n_drop, n, seed, otherchoice, picknum, picknum_no):
        probs = self.net.predict_prob_MarginSampling_dropout(data, n_drop, n, seed, otherchoice, picknum, picknum_no)
        return probs

    def predict_prob_EntropySampling_dropout(self, data, n_drop, n, seed, otherchoice, picknum, picknum_no):
        probs = self.net.predict_prob_EntropySampling_dropout(data, n_drop, n, seed, otherchoice, picknum)
        return probs

    def get_embeddings(self, data):
        embeddings = self.net.get_embeddings(data)
        return embeddings
