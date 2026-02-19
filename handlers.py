import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from image_tools import norm, faultseg_augumentation


    
class FAULTSEG_Handler(Dataset):
    def __init__(self, X, Y, isTrain):
        self.X = X
        self.Y = Y
        self.isTrain = isTrain


    def transform(self, img, mask):
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        img=norm(img)
        img = TF.normalize(img, [2.69254e-05,],[0.1701577, ])
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


        x,y=self.transform(x,y)
        return x, y, index

    def __len__(self):
        return len(self.X)
   
class THEBE_Handler(Dataset):
    def __init__(self, X, Y, isTrain):
        self.X = X
        self.Y = Y
        self.isTrain = isTrain


    def transform(self, img, mask):
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        img=norm(img)
        img = TF.normalize(img, [-3.26645e-05, ],[0.03790, ])
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
   
