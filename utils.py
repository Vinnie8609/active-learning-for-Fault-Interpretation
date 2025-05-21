from torchvision import transforms
from handlers import THEBE_Handler,FAULTSEG_Handler
from data import get_THEBE,get_FAULTSEG

from nets_test_transunet import Net, MNIST_Net, SVHN_Net, CIFAR10_Net,THEBE_Net,FAULTSEG_Net
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                             LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                             KMeansSampling, KCenterGreedy, BALDDropout, \
                             AdversarialBIM, AdversarialDeepFool,Orderselect,MaxconfidenceSampling

params = {'MNIST':
              {'n_epoch': 10, 
               'train_args':{'batch_size': 64, 'num_workers': 1},
               'test_args':{'batch_size': 1000, 'num_workers': 1},
               'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
          'FashionMNIST':
              {'n_epoch': 10, 
               'train_args':{'batch_size': 64, 'num_workers': 1},
               'test_args':{'batch_size': 1000, 'num_workers': 1},
               'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
          'SVHN':
              {'n_epoch': 20, 
               'train_args':{'batch_size': 64, 'num_workers': 1},
               'test_args':{'batch_size': 1000, 'num_workers': 1},
               'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
          'CIFAR10':
              {'n_epoch': 20, 
               'train_args':{'batch_size': 64, 'num_workers': 1},
               'test_args':{'batch_size': 1000, 'num_workers': 1},
               'optimizer_args':{'lr': 0.05, 'momentum': 0.3}},
          'FAULTSEG':
              {'n_epoch': 100, 
               'train_args':{'batch_size':16, 'num_workers': 4},
               'val_args':{'batch_size': 8, 'num_workers': 4},
               'test_args':{'batch_size': 4, 'num_workers': 4},#50
               'optimizer_args':{'lr': 0.002, 'momentum': 0.9}},
           'THEBE':
              {'n_epoch': 50, 
               'train_args':{'batch_size':8, 'num_workers': 0},
               'trainsmall_args':{'batch_size':2, 'num_workers': 0},
               'val_args':{'batch_size': 8, 'num_workers': 0},
               'test_args':{'batch_size':8 , 'num_workers': 0},#50
               'optimizer_args':{'lr': 0.002, 'momentum': 0.9}}
          }

def get_handler(name):

    if name == 'THEBE':
        return THEBE_Handler
    elif name == 'FAULTSEG':
        return FAULTSEG_Handler
    
def get_dataset(name):
    if name == 'THEBE':
        return get_THEBE(get_handler(name))
    elif name == 'FAULTSEG':
        return get_FAULTSEG(get_handler(name))
    else:
        raise NotImplementedError
        
def get_net(name, device):
    if name == 'MNIST':
        return Net(MNIST_Net, params[name], device)
    elif name == 'FashionMNIST':
        return Net(MNIST_Net, params[name], device)
    elif name == 'SVHN':
        return Net(SVHN_Net, params[name], device)
    elif name == 'CIFAR10':
        return Net(CIFAR10_Net, params[name], device)
    elif name == 'THEBE':
        return Net(THEBE_Net, params[name], device)
    elif name == 'FAULTSEG':
        return Net(FAULTSEG_Net, params[name], device)
    else:
        raise NotImplementedError
    
def get_params(name):
    return params[name]

def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "LeastConfidence":
        return LeastConfidence
    elif name == "MarginSampling":
        return MarginSampling
    elif name == "EntropySampling":
        return EntropySampling
    elif name == "LeastConfidenceDropout":
        return LeastConfidenceDropout
    elif name == "MarginSamplingDropout":
        return MarginSamplingDropout
    elif name == "EntropySamplingDropout":
        return EntropySamplingDropout
    elif name == "KMeansSampling":
        return KMeansSampling
    elif name == "KCenterGreedy":
        return KCenterGreedy
    elif name == "BALDDropout":
        return BALDDropout
    elif name == "AdversarialBIM":
        return AdversarialBIM
    elif name == "AdversarialDeepFool":
        return AdversarialDeepFool
    
    elif name == "Orderselect":
        return Orderselect
    elif name == "MaxconfidenceSampling":
        return MaxconfidenceSampling
    else:
        raise NotImplementedError
    
# albl_list = [MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args),
#              KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)]
# strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)
