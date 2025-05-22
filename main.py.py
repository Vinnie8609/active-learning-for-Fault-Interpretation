import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint
from common_tools import create_logger 
import os
import random
from model_predict_thebe import model_predict

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=456, help="random seed")    #111111111
parser.add_argument('--picknum', type=int, default=50, help="random seed")  
parser.add_argument('--otherchoice', type=str, default="transunt_3", help="number of round pick samples")    #30pices
parser.add_argument('--n_init_labeled', type=int, default=348, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=50, help="number of queries per round")
parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
parser.add_argument('--dataset_name', type=str, default="THEBE", choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10","THEBE","FAULTSEG"], help="dataset")
parser.add_argument('--strategy_name', type=str, default="EntropySampling",
                    choices=["Orderselect",
                            "RandomSampling", 
                             "LeastConfidence", 
                             "MarginSampling", 
                             "EntropySampling", 
                             "LeastConfidenceDropout", 
                             "MarginSamplingDropout", 
                             "EntropySamplingDropout", 
                             "KMeansSampling",
                             "KCenterGreedy", 
                             "BALDDropout", 
                             "AdversarialBIM", 
                             "AdversarialDeepFool"], help="query strategy")
args = parser.parse_args()
pprint(vars(args))
print()
 ################创建文件夹


if not os.path.exists("./active_learning_data/{}_{}".format(args.seed,args.otherchoice)):

    os.makedirs("./active_learning_data/{}_{}".format(args.seed,args.otherchoice))

os.makedirs("./active_learning_data/{}_{}/{}".format(args.seed,args.otherchoice,args.strategy_name))
os.makedirs("./active_learning_data/{}_{}/{}/log".format(args.seed,args.otherchoice,args.strategy_name))
os.makedirs("./active_learning_data/{}_{}/{}/predick_result".format(args.seed,args.otherchoice,args.strategy_name))
os.makedirs("./active_learning_data/{}_{}/{}/pick".format(args.seed,args.otherchoice,args.strategy_name))
os.makedirs("./active_learning_data/{}_{}/{}/picture".format(args.seed,args.otherchoice,args.strategy_name))
os.makedirs("./active_learning_data/{}_{}/{}/picture/test".format(args.seed,args.otherchoice,args.strategy_name))
os.makedirs("./active_learning_data/{}_{}/{}/picture/val".format(args.seed,args.otherchoice,args.strategy_name))

logger = create_logger("./active_learning_data/{}_{}/{}/log".format(args.seed,args.otherchoice,args.strategy_name),"main")





logger.info(args)

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
setup_seed(args.seed)
torch.backends.cudnn.enabled = False

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dataset = get_dataset(args.dataset_name)                   # load dataset
net = get_net(args.dataset_name, device)                   # load network
strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy

# start experiment
# dataset.initialize_labels(args.n_init_labeled)
print(f"number of labeled pool: {args.n_init_labeled}")
# print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
print(f"number of testing pool: {dataset.n_test}")
print()

logger.info(f"number of labeled pool: {args.n_init_labeled}")
# logger.info(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
logger.info(f"number of testing pool: {dataset.n_test}")

# # round 0 accuracy



# print("Round 0")
# logger.info("Round 0")
# query_idxs = strategy.query(0)


best_iou=strategy.train_before(0,args.strategy_name,args.seed,args.otherchoice)  
# flag=np.ones([15,512,2048],type="bool")
flag=np.ones([15])


# ############################################################

for rd in range(1, args.n_round+1):

    print(f"Round {rd}")
    logger.info(f"Round {rd}")

    # query
    flag_update = strategy.query(rd,args.seed,args.otherchoice,args.picknum,args.picknum_no,flag)#n_query 10 
    flag=flag_update 
    # strategy.update(query_idxs)
    a=strategy.train(rd,args.strategy_name,best_iou,args.seed,args.otherchoice)
   
    best_iou=a
    print(best_iou)
model_predict(rd)



    
