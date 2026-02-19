import argparse
import os
import random
from pprint import pprint

import numpy as np
import torch

from common_tools import create_logger
from model_predict_thebe import model_predict
from utils import get_dataset, get_net, get_strategy

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help="random seed")
parser.add_argument('--picknum', type=int, default=50, help="random seed")
parser.add_argument('--otherchoice', type=str, default="transunt_3", help="number of round pick samples")
parser.add_argument('--n_init_labeled', type=int, default=348, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=50, help="number of queries per round")
parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
parser.add_argument('--dataset_name', type=str, default="THEBE", help="dataset")
parser.add_argument('--strategy_name', type=str, default="EntropySampling", help="query strategy")
args = parser.parse_args()
pprint(vars(args))
print()
 ################创建文件夹


os.makedirs("./active_learning_data/{}_{}".format(args.seed,args.otherchoice), exist_ok=True)


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
print(f"number of labeled pool: {args.n_init_labeled}")
print(f"number of testing pool: {dataset.n_test}")
print()

logger.info(f"number of labeled pool: {args.n_init_labeled}")
logger.info(f"number of testing pool: {dataset.n_test}")

flag=np.ones([15])


# ############################################################

for rd in range(1, args.n_round+1):

    print(f"Round {rd}")
    logger.info(f"Round {rd}")

    # query
    if args.strategy_name == "EntropySampling":
        flag_update = strategy.query(rd, args.seed, args.otherchoice, flag)
    else:
        flag_update = strategy.query(rd, args.seed, args.otherchoice, args.picknum, flag)
    flag=flag_update
    a=strategy.train(rd,args.strategy_name,best_iou,args.seed,args.otherchoice)
   
    best_iou=a
    print(best_iou)
model_predict(rd)



    
