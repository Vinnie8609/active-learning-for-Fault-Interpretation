import numpy as np
import torch
from torchvision import datasets

class Data:
    def __init__(self, X_train_first, Y_train_first,X_train_middle, Y_train_middle,X_train_small, Y_train_small, X_val, Y_val,X_test, Y_test, handler):
        self.X_train_first = X_train_first
        self.Y_train_first = Y_train_first
        self.X_train_middle = X_train_middle
        self.Y_train_middle = Y_train_middle
        self.X_train_small = X_train_small
        self.Y_train_small = Y_train_small
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler
        
        self.n_pool = len(X_train_first)   #4000   100
        self.n_test = len(X_test)   #1000
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)   #4000 
        
    def initialize_labels(self, num):   #num=1000
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
        # print(self.labeled_idxs[tmp_idxs[:num]])
    
    def get_labeled_data(self):
        # labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        # print("aaaa",self.labeled_idxs)
        train_img=self.X_train_first
        print(train_img.shape)
        print(self.X_train_small.shape)
        train_mask=self.Y_train_first
        train_imgs_small=np.load("./data/THEBE224/val_img.npy")
        train_imgs_small=torch.tensor(train_imgs_small)
        
        # train_masks_small=np.load("/home/user/data/liuyue/active_learning/data/THEBE_NEW/train_mask_small.npy")
        train_masks_small=np.load("./data/THEBE224/val_img.npy")
        train_masks_small=torch.tensor(train_masks_small)
        print(train_imgs_small.shape)
        # if torch.sum(train_imgs_small)==0:
        new_x_train=train_img
        new_y_train=train_mask
        # else:
        #     new_x_train=torch.cat((train_img, train_imgs_small), dim=0)
        #     new_y_train=torch.cat((train_mask, train_masks_small), dim=0)
        return  self.handler(new_x_train, new_y_train,True)
    
   
   


    def get_unlabeled_data(self):
        unlabeled_idxs = np.arange(1)
        return unlabeled_idxs, self.handler(self.X_train_middle, self.Y_train_middle,True)
    
    def get_train_data(self):
        # print("aaaa",self.labeled_idxs)
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train,True)
    
    def get_val_data(self):
        return self.handler(self.X_val, self.Y_val,False)
        
    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test,False)
    
    
    def cal_test_acc(self, preds):
        # return 1.0 * (self.Y_test==preds).sum().item() / self.n_test
       pass



    
        
    



def get_THEBE(handler):

    img=np.load("./data/seistrain.npy")
    mask=np.load("./data/faulttrain.npy")
    num_frames = train_images.shape[0]
    selected_indices = np.random.choice(num_frames, size=25, replace=False)
    selected_images = img[selected_indices]
    selected_masks = mask[selected_indices]
    label_img= selected_images [:10,100:2148,700:1212]
    unlabel_img= selected_images [10:,100:2148,700:1212]
    label_mask= selected_masks [:10,100:2148,700:1212]
    unlabel_mask= selected_masks [10:,100:2148,700:1212]
    
    trainimg=np.zeros([180,224,224])
    trainmask=np.zeros([180,224,224])
    id=0
    for k in range(10):
        for i in range (9):
            for j in range(2):
                img1=label_img[k,i*224:i*224+224,j*224:j*224+224]
                mask1=label_mask[k,i*224:i*224+224,j*224:j*224+224]
                if mask1.sum!=0:
                    trainimg[id]=img1
                    trainmask[id]=mask1
                    id+=1
    trainimg=trainimg[:id]
    trainmask=trainmask[:id]

    np.save("./data/trainimg.npy",trainimg)
    np.save("./data/trainmask.npy",trainmask)
    np.save("./data/trainimg_unlabel",unlabel_img)
    np.save("./data/trainmask_unlabel",unlabel_mask)
    
    train_imgs_first=np.load("./data/trainimg.npy") # 348*128*128
   
    train_imgs_first=torch.tensor(train_imgs_first)
    
    train_masks_first=np.load("./data/trainmask.npy")
   
    train_masks_first=torch.tensor(train_masks_first)
    
    
    
    train_imgs_middle=np.load("./data/trainimg_unlabel.npy")
    train_imgs_middle=torch.tensor(train_imgs_middle)
    
    
    train_masks_middle=np.load("./data/trainmask_unlabel.npy")  
    train_masks_middle=torch.tensor(train_masks_middle)

    trainimg_small=np.zeros([50,224,224])
    trainmask_small=np.zeros([50,224,224])
    np.save("./data/trainimg_small.npy",trainimg_small)
    np.save("./data/trainmask_small.npy",trainmask_small)
   
    train_imgs_small=np.load("./data/trainimg_small.npy")
    train_imgs_small=torch.tensor(train_imgs_small)

    train_masks_small=np.load("./data//trainmask_small.npy")
    train_masks_small=torch.tensor(train_masks_small)

    img=np.load("./data/seisval.npy")
    mask=np.load("./data/faultval.npy")
    num_frames = train_images.shape[0]
    selected_indices = np.random.choice(num_frames, size=10, replace=False)
    selected_images = img[selected_indices]
    selected_masks = mask[selected_indices]
    label_img= selected_images [:,100:2148,700:1212]
    label_mask= selected_masks [:,100:2148,700:1212]
    
    
    valimg=np.zeros([180,224,224])
    valmask=np.zeros([180,224,224])
    id=0
    for k in range(10):
        for i in range (9):
            for j in range(2):
                img1=label_img[k,i*224:i*224+224,j*224:j*224+224]
                mask1=label_mask[k,i*224:i*224+224,j*224:j*224+224]
                if mask1.sum!=0:
                    valimg[id]=img1
                    valmask[id]=mask1
                    id+=1
    valimg=valimg[:id]
    valmask=valmask[:id]

    np.save("./data/valimg.npy",valimg)
    np.save("./data/valmask.npy",valmask)
    
    val_imgs=np.load("./data/valimg.npy")
    
    val_imgs=torch.tensor(val_imgs)

    val_masks=np.load("./data/valmask.npy")
    
    val_masks=torch.tensor(val_masks)


    
    test_imgs=np.load("./data/seistest.npy")
    
    test_imgs=torch.tensor(test_imgs)

   
    test_masks=np.load("./data/faulttest.npy")
  
    test_masks=torch.tensor(test_masks)
  

    return Data(train_imgs_first,train_masks_first, train_imgs_middle,train_masks_middle,train_imgs_small,train_masks_small, val_imgs, val_masks, test_imgs, test_masks, handler)
  
  
  


def get_FAULTSEG(handler):
    train_imgs=np.load("./data/faultseg/train/seis/train_img.npy")

    train_imgs=torch.tensor(train_imgs)
    
    train_masks=np.load("./data/faultseg/train/fault/train_mask.npy")
      
    train_masks=torch.tensor(train_masks)


    val_imgs=np.load("./data/faultseg/validation/seis/val_img.npy")
     
    val_imgs=torch.tensor(val_imgs)

    val_masks=np.load("./data/faultseg/validation/fault/val_mask.npy")
    
    val_masks=torch.tensor(val_masks)



    test_imgs=np.load("./data/THEBE/test_imgs.npy")
     
    test_imgs=torch.tensor(test_imgs)

    test_masks=np.load("./data/THEBE/test_masks.npy")
    
    test_masks=torch.tensor(test_masks)
  


    return Data(train_imgs[:1000],train_masks[:1000], val_imgs[:40], val_masks[:40], test_imgs[:20], test_masks[:20], handler)
