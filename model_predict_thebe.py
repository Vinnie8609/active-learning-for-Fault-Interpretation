
import argparse
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from common_tools import create_logger
from image_tools import *
from predictTimeSlice_transunet import *
from evalution_segmentaion import Evaluator

batch_size = 32

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=456, help="random seed")
parser.add_argument('--otherchoice', type=str, default="transunt_3", help="number of round pick samples")
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
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)     # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

setup_seed(args.seed)
torch.backends.cudnn.enabled = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
strategy_name = args.strategy_name
seed = args.seed
otherchoice = args.otherchoice

def model_predict(rd):
    logger = create_logger("./active_learning_data/{}_{}/{}/log".format(seed,otherchoice,strategy_name),"predict_{}".format(rd))
    seis = np.load("./data/THEBE_NEW/test_img_all.npy")
    
    t1 = time.time()
    for i in range(len(seis)):

       
        recover_Y_test_pred=predict_slice(THEBE_Net, seis[i],strategy_name,seed,otherchoice)
        np.save("./active_learning_data/{}_{}/{}/predick_result/{}.npy".format(seed,otherchoice,strategy_name,i),
                np.squeeze(recover_Y_test_pred))

    t2 = time.time()
    print('save in {} sec'.format(t2 - t1))

    miou = Evaluator(2)
    miouVal = 0
    accVal = 0
    mF1=0
    mRecall=0
    mPrecious=0
    num=len(seis)

    fault=np.load("./data/THEBE_NEW/test_mask_all.npy")
   
    for i in range(len(seis)):
            predicted_mask=np.load("./active_learning_data/{}_{}/{}/predick_result/{}.npy".format(seed,otherchoice,strategy_name,i))
            predicted_mask= predicted_mask>0.5
            img1=predicted_mask
            plt.figure(figsize=(4,4))
            plt.imshow(img1)
            
            plt.savefig("./active_learning_data/{}_{}/{}/picture/test/{}.png".format(seed,otherchoice,strategy_name,i))
                    
            mask=fault[i]
        #     print(fault[i])
            plt.figure(figsize=(4,4))
            plt.imshow(mask)
            
            plt.savefig("./active_learning_data/{}_{}/{}/picture/test/{}_maskS.png".format(seed,otherchoice,strategy_name,i))
                
                
            miou.add_batch(fault[i].astype(int), predicted_mask.astype(int))
            accVal += miou.Pixel_Accuracy()
            miouVal += miou.Mean_Intersection_over_Union()
            mRecall+=miou.Mean_Recall()
            mF1+=miou.Mean_F1()
            mPrecious+=miou.Precious
    accVal=accVal*100/ num
    accVal=('%.2f' % accVal)
    miouVal = miouVal*100/num
    miouVal = ('%.2f' % miouVal)
    mF1 = mF1*100/num
    mF1 = ('%.2f' % mF1)
    mPrecious = mPrecious * 100 / num
    mPrecious = ('%.2f' % mPrecious)
    mRecall= mRecall * 100 / num
    mRecall = ('%.2f' % mRecall)
    logger.info('round{}:  all acc:{} , miou:{}  , Precious:{}  ,Recall:{}   ,F1 :{} '.format(rd,accVal,miouVal, mPrecious,mRecall,mF1))

if __name__ == '__main__':
     rd=0
     model_predict(0)



