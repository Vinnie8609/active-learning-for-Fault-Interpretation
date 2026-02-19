from handlers import THEBE_Handler,FAULTSEG_Handler
from data import get_THEBE,get_FAULTSEG

from nets_test_transunet import Net,THEBE_Net,FAULTSEG_Net
from query_strategies import  EntropySampling
                            

params = {
           'THEBE':
              {'n_epoch': 10, 
               'train_args':{'batch_size':8, 'num_workers': 0},
               'trainsmall_args':{'batch_size':16, 'num_workers': 0},
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

    if name == 'THEBE':
        return Net(THEBE_Net, params[name], device)
    elif name == 'FAULTSEG':
        return Net(FAULTSEG_Net, params[name], device)
    else:
        raise NotImplementedError
    
def get_params(name):
    return params[name]

def get_strategy(name):

    if name == "EntropySampling":
        return EntropySampling

     
    else:
        raise NotImplementedError
    
