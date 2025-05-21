import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


from os.path import splitext
from os import listdir
from glob import glob
import numpy as np
import torchvision.transforms.functional as TF
from torch.nn import functional as F
from image_tools import norm,faultseg_augumentation


    
class FAULTSEG_Handler(Dataset):
    def __init__(self, X, Y,isTrain):
        self.X = X
        self.Y = Y
        self.isTrain=isTrain
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])


    def transform(self, img, mask):
        # to tensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        img=norm(img)
        img = TF.normalize(img, [2.69254e-05,],[0.1701577, ])############## [0.4915, ], [0.0655, ], mean=0.5   std=0.5
        return img, mask

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = np.asarray(x, dtype=np.float32)
        y= np.asarray(y, dtype=np.float32)

        if self.isTrain:  # 训练集，数据增强
            aug = faultseg_augumentation(p=0.7)

            augmented = aug(image=x, mask=y)
            x = augmented['image']
            y = augmented['mask']


        # x = Image.fromarray(x.numpy(), mode='L')
        x,y=self.transform(x,y)
        return x, y, index

    def __len__(self):
        return len(self.X)
   
class THEBE_Handler(Dataset):
    def __init__(self, X, Y,isTrain):
        self.X = X
        self.Y = Y
        self.isTrain=isTrain
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])


    def transform(self, img, mask):
        # to tensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        img=norm(img)
        img = TF.normalize(img, [-3.26645e-05, ],[0.03790, ])############## [0.4915, ], [0.0655, ]      [0.000384, ],[1.05163, ]
        return img, mask

    def __getitem__(self, index):
        # print(index)
        x, y = self.X[index], self.Y[index]
        x = np.asarray(x, dtype=np.float32)
        y= np.asarray(y, dtype=np.float32)

        if self.isTrain:  # 训练集，数据增强
            aug = faultseg_augumentation(p=0.7)

            augmented = aug(image=x, mask=y)
            x = augmented['image']
            y = augmented['mask']


        # x = Image.fromarray(x.numpy(), mode='L')
        x,y=self.transform(x,y)
        return x, y, index

    def __len__(self):
        return len(self.X)
   
