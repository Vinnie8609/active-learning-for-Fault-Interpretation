
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict
from scheduler import GradualWarmupScheduler
from common_tools import create_logger 
import losses
import os
import cv2
import cmapy
import matplotlib.pyplot as plt
from evalution_segmentaion import Evaluator

import torchvision.transforms.functional as TF

from image_tools import *
# from predictTimeSlice import predict_slice

import torch.utils.data
import time

from evalution_segmentaion import Evaluator
import copy
import logging
import math
from configs.config import get_config
from os.path import join as pjoin

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from sklearn.cluster import KMeans


import timm.models.vision_transformer
# from predictTimeSlice import *
from predictTimeSlice_transunet import *

from TransUnet import VisionTransformer
import TransUnet_vit_seg_configs as configs

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}



class Net:
    def __init__(self, net, params, device):
        self.net = net
        self.params = params
        self.device = device

    def train_before(self, train_data,val_data,n,strategy_name,seed,otherchoice):
        logger = create_logger("./active_learning_data/{}_{}/{}/log".format(seed,otherchoice,strategy_name),"train_{}".format(n))
        vit_name="R50-ViT-B_16"
        img_size=224
        vit_patches_size=16
        config_vit = CONFIGS[vit_name]
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (
            int(img_size / vit_patches_size), int(img_size / vit_patches_size))

        self.clf=VisionTransformer(config_vit).to(self.device)
        # self.clf.load_from(weights=np.load(config_vit.pretrained_path))
        

        best_miou=0
        
        

        criterion = torch.nn.CrossEntropyLoss()
        dice_loss = losses.DiceLoss(2)
        mse_loss=nn.MSELoss()
        
         

        n_epoch = self.params['n_epoch']
        
        
        mean_train_losses = []
        mean_val_losses = []
        mean_train_accuracies = []
        mean_val_accuracies = []
        
        

        
       
        optimizer = optim.AdamW(self.clf.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, 
                                weight_decay=0.001)    #0.0004
       
        # 定义 Warmup 学习率策略
        warmup_epochs = 10
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1
        )

        # 定义余弦退火学习率策略
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100 - warmup_epochs,eta_min=1e-6)
        


        train_loader = DataLoader(train_data, shuffle=True, **self.params['train_args'])
        val_loader = DataLoader(val_data, shuffle=False, **self.params['val_args'])
        
        for epoch in tqdm(range(1, n_epoch+1), ncols=100):
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []
            self.clf.train()
            for batch_idx, (x, y, idxs) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                out = self.clf(x)  #16,2,128,128
                outputs=torch.zeros([out.size(0),2,224,224])
                outputs[:,1,:,:]=out.squeeze(1)
                outputs[:,0,:,:]=1-out.squeeze(1)
                predicted_mask = out > 0.5
               
                tloss_ce = criterion(outputs.to(self.device),y.squeeze(1).long())
                tloss_dice = dice_loss(outputs.to(self.device), y)
                
                tloss_mse=mse_loss(outputs[:,0,:,:].to(self.device),1-y)
                
                tloss=tloss_ce+tloss_dice+ tloss_mse
                logger.info("Epoch {}: tloss_ce: {:.4f},tloss_mse:{:.4f},,tloss_dice: {:.4f}".format(epoch, tloss_ce.item(),tloss_mse.item(),tloss_dice.item()))#
                
                
               
                tloss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_losses.append(tloss.data)
                
                train_acc = iou_pytorch(predicted_mask.squeeze(1).byte(), y.squeeze(1).byte(),1e-6)
                train_accuracies.append(train_acc.mean())
                
                logger.info("Epoch {}: Acc: {:.2%},Loss: {:.4f}".format(epoch, 
                                                                            train_acc.mean().item(),tloss.item()))
            
            if epoch < warmup_epochs:
                    warmup_scheduler.step()
            else:
                    cosine_scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1}, Learning Rate: {current_lr}")

          
            self.clf.eval()
            for x, y, idxs in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)  #16,1,224,224
                
                outputs=torch.zeros([out.size(0),2,224,224])
                outputs[:,1,:,:]=out.squeeze(1)
                outputs[:,0,:,:]=1-out.squeeze(1)
                
                predicted_mask = out > 0.5
                vloss_ce = criterion(outputs.to(self.device),y.squeeze(1).long())
                vloss_dice = dice_loss(outputs.to(self.device), y)
                
                vloss_mse=mse_loss(outputs[:,0,:,:].to(self.device),1-y)
                
                vloss=vloss_ce+vloss_dice+vloss_mse
                
                logger.info("Epoch {}: vloss_ce: {:.4f},vloss_mse:{:.4f},vloss_dice: {:.6f}".format(epoch, vloss_ce.item(),vloss_mse.item(),vloss_dice.item()))#
                
                val_losses.append(vloss.data)
                val_acc = iou_pytorch(predicted_mask.squeeze(1).byte(), y.squeeze(1).byte(),1e-6)
                logger.info("idx {}: Acc: {:.2%},loss:{}".format(idxs, val_acc.mean().item(),vloss.mean()))
                val_accuracies.append(val_acc.mean())
            
            mean_train_losses.append(torch.mean(torch.stack(train_losses)))
            mean_val_losses.append(torch.mean(torch.stack(val_losses)))
            mean_train_accuracies.append(torch.mean(torch.stack(train_accuracies)))
            mean_val_accuracies.append(torch.mean(torch.stack(val_accuracies)))
            val_iou = torch.mean(torch.stack(val_accuracies))    
            logger.info('Epoch: {}. Train Loss: {:.4f}. Val Loss: {:.4f}. Train IoU: {:.4f}. Val IoU: {:.4f}. '
                .format(epoch , torch.mean(torch.stack(train_losses)), torch.mean(torch.stack(val_losses)),
                        torch.mean(torch.stack(train_accuracies)),val_iou))
            
            if best_miou < val_iou.item() :

                best_miou = val_iou.item() 
                checkpoint = {"model_state_dict": self.clf.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_miou": best_miou}
                pkl_name = "SSL_checkpoint_best.pkl"

                

                path_checkpoint = os.path.join("./active_learning_data/{}_{}/{}".format(seed,otherchoice,strategy_name), pkl_name)
                torch.save(checkpoint, path_checkpoint)
                logger.info("best_miou is :{}".format(best_miou))

                
            # if epoch==100:
                img=predicted_mask.squeeze(1)[0,:,:].cpu()
                plt.imshow(img)
                plt.savefig("./active_learning_data/{}_{}/{}/picture/val/{}_{}.png".format(seed,otherchoice,strategy_name,n,int(idxs[0])))

                mask=y.squeeze(1)[0,:,:].cpu()
                plt.imshow(mask)
                plt.savefig("./active_learning_data/{}_{}/{}/picture/val/{}_{}_mask.png".format(seed,otherchoice,strategy_name,n,int(idxs[0])))
        # print (best_miou)
        return best_miou 


    def train(self, train_data,val_data,n,strategy_name,best_iou,seed,otherchoice):
        logger = create_logger("./data/liuyue/active_learning_data/{}_{}/{}/log".format(seed,otherchoice,strategy_name),"train_{}".format(n))
        
        vit_name="R50-ViT-B_16"
        img_size=224
        vit_patches_size=16
        config_vit = CONFIGS[vit_name]
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (
            int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        self.clf=VisionTransformer(config_vit).to(self.device)
        model_nestunet_path = "./active_learning_data/{}_{}/{}/SSL_checkpoint_best.pkl".format(seed,otherchoice,strategy_name)
        weights = torch.load(model_nestunet_path, map_location="cuda")['model_state_dict']
        weights_dict = {}
        for k, v in weights.items():
                new_k = k.replace('module.', '') if 'module' in k else k
                weights_dict[new_k] = v
        self.clf.load_state_dict(weights_dict)

        best_miou=best_iou
        print(best_iou)
        print(best_miou)
        

        criterion = torch.nn.CrossEntropyLoss()
        dice_loss = losses.DiceLoss(2)
        mse_loss=nn.MSELoss()
        

        n_epoch = self.params['n_epoch']
        
        
        mean_train_losses = []
        mean_val_losses = []
        mean_train_accuracies = []
        mean_val_accuracies = []
        
        

        
        # optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])
        optimizer = optim.AdamW(self.clf.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=0.001)
        # optimizer = optim.Adam(self.clf.parameters(), lr=0.00001,eps=1e-4)
        
        # 定义 Warmup 学习率策略
        warmup_epochs = 10
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1
        )

        # 定义余弦退火学习率策略
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100 - warmup_epochs,eta_min=1e-6)


        


        train_loader = DataLoader(train_data, shuffle=True, **self.params['train_args'])
        val_loader = DataLoader(val_data, shuffle=False, **self.params['val_args'])
        # trloss=[]
        # # my_list = list(range(100))
        # xlable=0
        for epoch in tqdm(range(1, n_epoch+1), ncols=100):
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []
            self.clf.train()
            for batch_idx, (x, y, idxs) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                out = self.clf(x)
                # print(out.cpu().shape)
                outputs=torch.zeros([out.size(0),2,224,224])
                outputs[:,1,:,:]=out.squeeze(1)
                outputs[:,0,:,:]=1-out.squeeze(1)
                
                predicted_mask = out > 0.5
                tloss_ce = criterion(outputs.to(self.device),y.squeeze(1).long())
                
                tloss_dice =dice_loss(outputs.to(self.device), y)
                
                tloss_mse=mse_loss(outputs[:,0,:,:].to(self.device),1-y)
               
                tloss=tloss_ce+tloss_dice+ tloss_mse
                logger.info("Epoch {}: tloss_ce: {:.4f},tloss_dice: {:.4f},tloss_mse:{:.4f}".format(epoch, tloss_ce.item(),tloss_dice.item(),tloss_mse.item()))#
                
                tloss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_losses.append(tloss.data)
                
                train_acc = iou_pytorch(predicted_mask.squeeze(1).byte(), y.squeeze(1).byte(),1e-6)
                train_accuracies.append(train_acc.mean())
                
                logger.info("Epoch {}: Acc: {:.2%},Loss: {:.4f}".format(epoch, 
                                                                            train_acc.mean().item(), tloss.item()))

            if epoch < warmup_epochs:
                    warmup_scheduler.step()
            else:
                    cosine_scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1}, Learning Rate: {current_lr}")

            # if epoch%10==0:
            self.clf.eval()
            for x, y, idxs in val_loader:
                x, y = x.to(self.device), y.to(self.device)
             
                out = self.clf(x)  #16,1,224,224
                
                outputs=torch.zeros([out.size(0),2,224,224])
                outputs[:,1,:,:]=out.squeeze(1)
                outputs[:,0,:,:]=1-out.squeeze(1)
                
                predicted_mask = out > 0.5
                vloss_ce = criterion(outputs.to(self.device),y.squeeze(1).long())
                
                
                vloss_mse=mse_loss(outputs[:,0,:,:].to(self.device),1-y)
                vloss_dice = dice_loss(outputs.to(self.device),y)
                
                vloss=vloss_ce+vloss_dice+vloss_mse
                logger.info("Epoch {}: vloss_ce: {:.4f},vloss_dice: {:.4f},vloss_mse:{:.4f}".format(epoch, vloss_ce.item(),vloss_dice.item(),vloss_mse.item()))#
                
                val_losses.append(vloss.data)
                val_acc = iou_pytorch(predicted_mask.squeeze(1).byte(), y.squeeze(1).byte(),1e-6)
                logger.info("idx {}: Acc: {:.2%},loss:{}".format(idxs, val_acc.mean().item(),vloss.mean()))
                val_accuracies.append(val_acc.mean())
            
            mean_train_losses.append(torch.mean(torch.stack(train_losses)))
            mean_val_losses.append(torch.mean(torch.stack(val_losses)))
            mean_train_accuracies.append(torch.mean(torch.stack(train_accuracies)))
            mean_val_accuracies.append(torch.mean(torch.stack(val_accuracies)))
            val_iou = torch.mean(torch.stack(val_accuracies))    
            logger.info('Epoch: {}. Train Loss: {:.4f}. Val Loss: {:.4f}. Train IoU: {:.4f}. Val IoU: {:.4f}. '
                .format(epoch , torch.mean(torch.stack(train_losses)), torch.mean(torch.stack(val_losses)),
                        torch.mean(torch.stack(train_accuracies)),val_iou))
            
            if best_miou < val_iou.item() :

                best_miou = val_iou.item() 
                checkpoint = {"model_state_dict": self.clf.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_miou": best_miou}
                pkl_name = "SSL_checkpoint_best.pkl"

                path_checkpoint = os.path.join("./active_learning_data/{}_{}/{}".format(seed,otherchoice,strategy_name), pkl_name)
                torch.save(checkpoint, path_checkpoint)
                logger.info("best_miou is :{}".format(best_miou))

                
            # if epoch==100:
                img=predicted_mask.squeeze(1)[0,:,:].cpu()
                plt.imshow(img)
                plt.savefig("./active_learning_data/{}_{}/{}/picture/val/{}_{}.png".format(seed,otherchoice,strategy_name,n,int(idxs[0])))

                mask=y.squeeze(1)[0,:,:].cpu()
                plt.imshow(mask)
                plt.savefig("./active_learning_data/{}_{}/{}/picture/val/{}_{}_mask.png".format(seed,otherchoice,strategy_name,n,int(idxs[0])))
        # print (best_miou)
        return best_miou


    
    

    def predict_prob_RandomSampling(self, data,n,seed,otherchoice,picknum,picknum_no,flag):
        loader = DataLoader(data, shuffle=False, **self.params['trainsmall_args'])
        for x, y, idxs in loader:
            x, y = x.to(self.device), y.to(self.device)
            new_train_imgs_small=np.zeros([picknum,224,224])
            new_train_masks_small=np.zeros([picknum,224,224])
            # fid=np.random.randint(64, 936, size=(50))
            # sid = np.random.randint(64, 1936, size=(50))
            fid=np.random.randint(112,400, size=(picknum))
            sid = np.random.randint(112, 1936, size=(picknum))

        # 找到所有值为 1 的索引
            indices = np.where(flag == 1)[0]  # np.where 返回的是元组，选择第一个元素

            # 从这些索引中随机选择一个索引
            random_index = np.random.choice(indices)
            flag[random_index]=0
            image=x.squeeze(1)[random_index].cpu()
            masks=y.squeeze(1)[random_index].cpu()
            maskContour=[]
            
            for i in range(picknum):
                firstid=fid[i]
                secondid=sid[i]
                maskContour.append((secondid,firstid))
                for j in range(224):
                    for z in range(224):
                        new_train_imgs_small[i][j][z]=image[(firstid-112+j)][(secondid-112+z)]
                        new_train_masks_small[i][j][z]=masks[(firstid-112+j)][(secondid-112+z)]
            

         # 克隆图像
        resultImg = masks.numpy().copy()*255
        resultImg= np.uint8(np.clip(resultImg, 0, 255))  # 限制范围在 [0, 255] 之间

        # 创建一个彩色图像
        m_resultImg = cv2.cvtColor(resultImg, cv2.COLOR_GRAY2BGR)

        # 遍历每个连通域点并绘制红色圆圈
        for point in maskContour:
            cv2.circle(m_resultImg, point, 1, (0, 0, 255), 10)  # 红色圆圈

    

        # 使用matplotlib显示图像
        # matplotlib默认是RGB格式，所以要将BGR格式转换为RGB
        m_resultImg_rgb = cv2.cvtColor(m_resultImg, cv2.COLOR_BGR2RGB)

        # 显示图像
        plt.imshow(m_resultImg_rgb)
        # plt.axis('off')  # 不显示坐标轴
        # plt.show()
        plt.savefig("./active_learning_data/{}_{}/{}/pick/split_{}/_mask_points.png".format(seed,otherchoice,"RandomSampling",n))


        if n==1:
                new_train_imgs= new_train_imgs_small
                new_train_masks=new_train_masks_small
                print("new_train_imgs:{}".format(new_train_imgs.shape))
                print("new_train_masks:{}".format(new_train_masks.shape))
        else:
                imgbefor=np.load("./active_learning/data/THEBE_224/train_img_small.npy")
                maskbefor=np.load("./active_learning/data/THEBE_224/train_mask_small.npy")
                print("imgbefor:{}".format(imgbefor.shape))
                print("maskbefor:{}".format(maskbefor.shape))
                imgbefor=torch.tensor(imgbefor)
                maskbefor=torch.tensor(maskbefor)
                new_train_imgs_small=torch.tensor(new_train_imgs_small)
                new_train_masks_small=torch.tensor(new_train_masks_small)
                new_train_imgs= torch.cat((new_train_imgs_small, imgbefor), dim=0)
                new_train_masks=torch.cat((new_train_masks_small, maskbefor), dim=0)
                print("new_train_imgs:{}".format(new_train_imgs.shape))
                print("new_train_masks:{}".format(new_train_masks.shape))    
        
        np.save("./data/THEBE_224/train_img_small.npy",new_train_imgs)
        np.save("./data/THEBE_224/train_mask_small.npy",new_train_masks)
        
        
           
        return random_index, flag 
      
    



    def predict_prob_MarginSampling(self, data,n,seed,otherchoice,picknum,picknum_no,flag):#最小   正常版本，，不变化
        vit_name="R50-ViT-B_16"
        img_size=224
        vit_patches_size=16
        config_vit = CONFIGS[vit_name]
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (
            int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        self.clf=VisionTransformer(config_vit).to(self.device)

        
        # self.clf = self.net().to(self.device)
        model_nestunet_path =  "./active_learning_data/{}_{}/{}/SSL_checkpoint_best.pkl".format(seed,otherchoice,"MarginSampling")
        weights = torch.load(model_nestunet_path, map_location="cuda")['model_state_dict']
        weights_dict = {}
        for k, v in weights.items():
                new_k = k.replace('module.', '') if 'module' in k else k
                weights_dict[new_k] = v
        self.clf.load_state_dict(weights_dict)

        self.clf.eval()
                

        loader = DataLoader(data, shuffle=False, **self.params['trainsmall_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs=np.zeros([len(idxs),2,512,2048])
        for idx in idxs:
            recover_Y_test_pred=predict_slice(THEBE_Net, x[idx].squeeze().cpu(),"MarginSampling",seed,otherchoice)#512,2048,1
            outputs[idx,1,:,:]=np.squeeze(recover_Y_test_pred)
        outputs[:,0,:,:]=1-outputs[:,1,:,:]
        outputs=torch.tensor(outputs)
        predict= torch.argmax(outputs,dim=1) 
        num=abs(outputs[:,1,:,:]-outputs[:,0,:,:])
        num=num.cpu()

        data={}
        for idx in idxs:
            if flag[idx]==1:
                points=min_50(num[idx])  #50个坐标
                # print(points)
                labels, centroids=kmeans(points)   #labels=0,1,2
                ####################################可视化 聚类的点
                plt.figure(figsize=(4,4))
                plt.scatter(points[:, 1], points[:, 0],s=10, c=labels, cmap='viridis')
                plt.scatter(centroids[:, 1], centroids[:, 0], s=20, c='red', marker='X')  # 绘制簇中心
                plt.title("K-means Clustering (K=3)")
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.savefig("./active_learning_data/{}_{}/{}/pick/split_{}/picture{}_kmeans.png".format(seed,otherchoice,"MarginSampling",n,idx))
                plt.close()

               
                #############################创建字典
                labels0=np.where(labels==0)   #索引值
                labels1=np.where(labels==1)
                labels2=np.where(labels==2)
                point0=points[labels0]  #对应的区域点
                point1=points[labels1]
                point2=points[labels2]
                area0=(point0[:,1].min(),point0[:,1].max())
                area1=(point1[:,1].min(),point1[:,1].max())
                area2=(point2[:,1].min(),point2[:,1].max())
                # count0=len(labels0[0])
                # count1=len(labels1[0])
                # count2=len(labels2[0])
                data["image_{}".format(idx)]={"area0": area0 ,"point0": point0 ,"count0":0,"area1":  area1,"point1":  point1 ,"count1":0,"area2": area2  ,"point2": point2,"count2":0}                        
               
                # data["image_{}".format(idx)]={"area0": area0 ,"point0": point0 ,"count0":count0,"area1":  area1,"point1":  point1 ,"count1":count1,"area2": area2  ,"point2": point2,"count2":count2}                        
                # print(data) 

                ####################################可视化 pridect
                # # 克隆图像
                resultImg =predict[idx,:,:].cpu().numpy().copy()*255
                resultImg= np.uint8(np.clip(resultImg, 0, 255))  # 限制范围在 [0, 255] 之间

                # 创建一个彩色图像
                m_resultImg = cv2.cvtColor(resultImg, cv2.COLOR_GRAY2BGR)

                # 遍历每个连通域点并绘制红色圆圈
                for point in point0:
                    point=tuple(point.tolist())
                    point_change=(point[1],point[0])
                    cv2.circle(m_resultImg, point_change, 1, (0, 0, 255), 15)  # 红色圆圈
                for point in point1:
                    point=tuple(point.tolist())
                    point_change=(point[1],point[0])
                    cv2.circle(m_resultImg, point_change, 1, (0, 255, 0), 15)  # 红色圆圈
                for point in point2:
                    point=tuple(point.tolist())
                    point_change=(point[1],point[0])
                    cv2.circle(m_resultImg, point_change, 1, (255, 255, 0), 15)  # 红色圆圈

                
                # 使用matplotlib显示图像
                # matplotlib默认是RGB格式，所以要将BGR格式转换为RGB
                m_resultImg_rgb = cv2.cvtColor(m_resultImg, cv2.COLOR_BGR2RGB)

                # 显示图像
                plt.figure(figsize=(4,4))
                plt.imshow(m_resultImg_rgb)
                # plt.axis('off')  # 不显示坐标轴
                # plt.show()
                plt.savefig("./active_learning_data/{}_{}/{}/pick/split_{}/picture{}_pridect_points.png".format(seed,otherchoice,"MarginSampling",n,idx))
                plt.close()
                
            else:
                    data["image_{}".format(idx)]={"area0":[] ,"point0": [] ,"count0":0,"area1":  [],"point1":  [] ,"count1":0,"area2": []  ,"point2": [],"count2":0}                        
               
                    continue
        # print(data)     
        # with open('/home/user/data/liuyue/active_learning_data/{}_{}/{}/pick/split_{}/data.txt'.format(seed,otherchoice,"MarginSampling",n), 'w') as f:
        #     json.dump(data, f)
        # ########################计算区域内标注的和
        area_sum=torch.zeros([len(idxs),3])
        for i in range(len(idxs)):
            for j in range(3):
                if flag[i]==1:
                    left=data["image_{}".format(i)]["area{}".format(j)][0]
                    right=data["image_{}".format(i)]["area{}".format(j)][1]
                    sum=torch.sum(predict[i,:,left:right])
                    area_sum[i,j]=sum
                    data["image_{}".format(i)]["count{}".format(j)]=sum

        #############################找到和最大的1个区域
        flattened_tensor = area_sum.flatten()
        # 2. 获取最大的 1 个元素的索引
        values, indices = torch.topk(flattened_tensor, 1, largest=True)
        # 3. 将一维索引转换为二维坐标
        # 使用 divmod 来获取行和列
        rows, cols = np.divmod(indices, area_sum.size(1))  # tensor.size(1) 是列数
        # 输出最大的 1个元素的坐标
        coordinates = torch.stack((rows, cols), dim=1)  #each_count中的位置坐标
        print(coordinates )     #a=tensor([[0, 0]])
        # int(a[0][1]) =0
        #############################找到点数最多的1个区域
        # each_count=torch.zeros([len(idxs),3])
        # for i in range(len(idxs)):
        #     for j in range(3):
        #             each_count[i,j]=data["image_{}".format(i)]["count{}".format(j)]
        # flattened_tensor = each_count.flatten()
        # # 2. 获取最大的 1 个元素的索引
        # values, indices = torch.topk(flattened_tensor, 1, largest=True)
        # # 3. 将一维索引转换为二维坐标
        # # 使用 divmod 来获取行和列
        # rows, cols = np.divmod(indices, each_count.size(1))  # tensor.size(1) 是列数
        # # 输出最大的 3个元素的坐标
        # coordinates = torch.stack((rows, cols), dim=1)  #each_count中的位置坐标
        # print(coordinates )     #a=tensor([[0, 0]])
        # # int(a[0][1]) =0
        ###############################找到点数最多的区域  -》标注
        num_image_pick=int(coordinates[0][0])
        num_area_pick=int(coordinates[0][1])
        flag[ num_image_pick]=0
        left=data["image_{}".format( num_image_pick)]["area{}".format(num_area_pick)][0]   
        right=data["image_{}".format( num_image_pick)]["area{}".format(num_area_pick)][1]   
        print(left,right)  #397,514
        img_area=x.squeeze(1)[num_image_pick][:,left:right]
        mask_area=y.squeeze(1)[num_image_pick][:,left:right]
    ###################################对选定的区域进行可视化
    # # 克隆图像
        resultImg =predict[num_image_pick,:,:].cpu().numpy().copy()*255
        resultImg= np.uint8(np.clip(resultImg, 0, 255))  # 限制范围在 [0, 255] 之间

        # 创建一个彩色图像
        m_resultImg = cv2.cvtColor(resultImg, cv2.COLOR_GRAY2BGR)

        #绘制所选中区域
        a=np.where( predict[num_image_pick,:,left:right]!=1)
        b=(a[1]+int(left),a[0])
        for i in range(b[0].size):
            d=(b[0][i],b[1][i])
            cv2.circle(m_resultImg, d, 1, (160,160,160), 1)

        # 绘制三个区域点
        for point in data["image_{}".format( num_image_pick)]["point0"]:
            point=tuple(point.tolist())
            point_change=(point[1],point[0])
            cv2.circle(m_resultImg, point_change, 1, (0, 0, 255), 15)  # 红色圆圈
        for point in data["image_{}".format( num_image_pick)]["point1"]:
            point=tuple(point.tolist())
            point_change=(point[1],point[0])
            cv2.circle(m_resultImg, point_change, 1, (0, 255, 0), 15)  # 绿色圆圈
        for point in data["image_{}".format( num_image_pick)]["point2"]:
            point=tuple(point.tolist())
            point_change=(point[1],point[0])
            cv2.circle(m_resultImg, point_change, 1, (255, 255, 0), 15)  # 黄色圆圈

        
        # 使用matplotlib显示图像
        # matplotlib默认是RGB格式，所以要将BGR格式转换为RGB
        m_resultImg_rgb = cv2.cvtColor(m_resultImg, cv2.COLOR_BGR2RGB)

        # 显示图像
        plt.figure(figsize=(4,4))
        plt.imshow(m_resultImg_rgb)
        # plt.axis('off')  # 不显示坐标轴
        # plt.show()
        plt.savefig("./active_learning_data/{}_{}/{}/pick/split_{}/picture{}_pickarea_pridect_points.png".format(seed,otherchoice,"MarginSampling",n,num_image_pick))
        plt.close()

        ###############################求这个区域的连通性
        con_nums=[]
        mask_area=mask_area.cpu().numpy().astype(np.uint8)
        # 连通性分析
        num_labels, labels = cv2.connectedComponents(mask_area, connectivity=8)
        output_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

        # 为每个连通组件指定不同的颜色
        for label in range(1, num_labels):  # 0 是背景，跳过
            con_nums.append(label)
            output_image[labels == label] = np.random.randint(0, 255, 3)

        # 显示图像
        plt.imshow(output_image)
        plt.axis('off')  # 不显示坐标轴
        plt.savefig("./active_learning_data/{}_{}/{}/pick/split_{}/area{}_connect.png".format(seed,otherchoice,"MarginSampling",n,num_area_pick))
        
        np.savetxt('./active_learning_data/{}_{}/{}/pick/split_{}/area{}_labels.txt'.format(seed,otherchoice,"MarginSampling",n,num_area_pick), labels, fmt='%d', delimiter=',')
        ######labels是一个矩阵，由0，1，2，3，，，，，类
        #########################根据连通性切割小图，
        new_train_imgs_small=torch.zeros([100,224,224])
        new_train_masks_small=torch.zeros([100,224,224])
        id=0
        #################按照断层连通性
        for label in range(1, num_labels):
            fids,sids=np.where(labels==label)
            # for ids in range(len(fids)):
            #     maskContour.append((sids[ids],fids[ids]))
            l=abs(fids[-1]-fids[0])
            w=abs(sids[-1]-sids[0])
            pickimg=np.zeros([l,w])
            pickmask=np.zeros([l,w])
            pickimg=img_area[min(fids[0],fids[-1]):max(fids[0],fids[-1]),min(sids[0],sids[-1]):max(sids[0],sids[-1])]
            pickmask=mask_area[min(fids[0],fids[-1]):max(fids[0],fids[-1]),min(sids[0],sids[-1]):max(sids[0],sids[-1])]
            plt.figure(figsize=(4,4))
            plt.imshow(pickimg.cpu())
            
            plt.savefig("./active_learning_data/{}_{}/{}/pick/split_{}/{}_img.png".format(seed,otherchoice,"MarginSampling",n,label))


            plt.figure(figsize=(4,4))
            plt.imshow(pickmask)
        
            plt.savefig("./active_learning_data/{}_{}/{}/pick/split_{}/{}_mask.png".format(seed,otherchoice,"MarginSampling",n,label))
        
            
            a=fids[0]
            b=sids[0]
            
            if (sids[-1]-sids[0] )>0:
                while(1):
                    # print(a,b)
                    if a>288 or b>(right-left-224):
                        break
                    new_train_imgs_small[id]=torch.tensor(img_area[a:a+224,b:b+224])
                    new_train_masks_small[id]=torch.tensor(mask_area[a:a+224,b:b+224])
                
                    c=np.where(fids==a+56)
                    if c[0].size==0:
                        break
                    else:
                        a=fids[c[0][0]]
                        b=sids[c[0][0]]
                        id+=1
                        # print(index)
                   

            else:
                while(1):
                    print(a,b)
                    if a>288 or b<224:
                        break
                    new_train_imgs_small[id]=torch.tensor(img_area[a:a+224,b-224:b])
                    new_train_masks_small[id]=torch.tensor(mask_area[a:a+224,b-224:b])
                    
                    c=np.where(fids==a+56)
                    if c[0].size==0:
                        break
                    else:
                        a=fids[c[0][0]]
                        b=sids[c[0][0]]
                        id+=1
        print(id)
        
        #################按照patch
        # for j in range(4):
            # for k in range(int(right-left)//128):
            #     img1=img_area[j*128:(j+1)*128,k*128:(k+1)*128]
            #     mask1=mask_area[j*128:(j+1)*128,k*128:(k+1)*128]
            #     if mask1.sum()!=0:
            #         new_train_imgs_small[id]=torch.tensor(img1)
            #         new_train_masks_small[id]=torch.tensor(mask1)
            #         id+=1
        # print(id)
        

        new_train_imgs_small=new_train_imgs_small[:id]
        new_train_masks_small=new_train_masks_small[:id]
        if n==1:
                new_train_imgs= new_train_imgs_small
                new_train_masks=new_train_masks_small
                print("new_train_imgs:{}".format(new_train_imgs.shape))
                print("new_train_masks:{}".format(new_train_masks.shape))
        else:
                imgbefor=np.load("./data/THEBE_224/train_img_small.npy")
                maskbefor=np.load("./data/THEBE_224/train_mask_small.npy")
                print("imgbefor:{}".format(imgbefor.shape))
                print("maskbefor:{}".format(maskbefor.shape))
                imgbefor=torch.tensor(imgbefor)
                maskbefor=torch.tensor(maskbefor)
                new_train_imgs_small=torch.tensor(new_train_imgs_small)
                new_train_masks_small=torch.tensor(new_train_masks_small)
                new_train_imgs= torch.cat((new_train_imgs_small, imgbefor), dim=0)
                new_train_masks=torch.cat((new_train_masks_small, maskbefor), dim=0)
                print("new_train_imgs:{}".format(new_train_imgs.shape))
                print("new_train_masks:{}".format(new_train_masks.shape))
        
        
    
        np.save("./data/THEBE_224/train_img_small.npy",new_train_imgs)
        np.save("./data/THEBE_224/train_mask_small.npy",new_train_masks)              
        # print(flag)
        return data,flag


    def predict_prob_EntropySampling(self, data,n,seed,otherchoice,picknum,picknum_no,flag):
        # self.clf = self.net().to(self.device)
        vit_name="R50-ViT-B_16"
        img_size=224
        vit_patches_size=16
        config_vit = CONFIGS[vit_name]
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (
            int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        self.clf=VisionTransformer(config_vit).to(self.device)
        model_nestunet_path = "./active_learning_data/{}_{}/{}/SSL_checkpoint_best.pkl".format(seed,otherchoice,"EntropySampling")
        weights = torch.load(model_nestunet_path, map_location="cuda")['model_state_dict']
        weights_dict = {}
        for k, v in weights.items():
                new_k = k.replace('module.', '') if 'module' in k else k
                weights_dict[new_k] = v
        self.clf.load_state_dict(weights_dict)

        self.clf.eval()
        
       
        
        loader = DataLoader(data, shuffle=False, **self.params['trainsmall_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
               
                outputs=np.zeros([len(idxs),2,512,2048])
        for idx in idxs:
            recover_Y_test_pred=predict_slice(THEBE_Net, x[idx].squeeze().cpu(),"EntropySampling",seed,otherchoice)#512,2048,1
            outputs[idx,1,:,:]=np.squeeze(recover_Y_test_pred)

        outputs[:,0,:,:]=1-outputs[:,1,:,:]
        outputs=torch.tensor(outputs)
        
        predict= torch.argmax(outputs,dim=1)             
        # print(predict.shape)
        num_0=outputs[:,0,:,:]
        num_1=outputs[:,1,:,:]
        num0_log=torch.log(num_0)
        num1_log=torch.log(num_1)
        num0_log = torch.nan_to_num(num0_log, nan=0.0)
        num1_log = torch.nan_to_num(num1_log, nan=0.0)
        entr0_log=num_0*num0_log
        entr1_log=num_1*num1_log
        
        entr_sum=-entr0_log-entr1_log
        entr_sum=entr_sum.cpu()

        # print(entr_sum.max())
        # flag=np.ones([len(idxs),512,2048],type="bool")
        data={}
        for idx in idxs:
            if flag[idx]==1:
                points=max_50(entr_sum[idx])  #50个坐标
                # print(points)
                labels, centroids=kmeans(points)   #labels=0,1,2
                ####################################可视化 聚类的点
                plt.figure(figsize=(4,4))
                plt.scatter(points[:, 1], points[:, 0],s=10, c=labels, cmap='viridis')
                plt.scatter(centroids[:, 1], centroids[:, 0], s=20, c='red', marker='X')  # 绘制簇中心
                plt.title("K-means Clustering (K=3)")
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.savefig("./active_learning_data/{}_{}/{}/pick/split_{}/picture{}_kmeans.png".format(seed,otherchoice,"EntropySampling",n,idx))
                plt.close()

               
                #############################创建字典
                labels0=np.where(labels==0)   #索引值
                labels1=np.where(labels==1)
                labels2=np.where(labels==2)
                point0=points[labels0]  #对应的区域点
                point1=points[labels1]
                point2=points[labels2]
                area0=(point0[:,1].min(),point0[:,1].max())
                area1=(point1[:,1].min(),point1[:,1].max())
                area2=(point2[:,1].min(),point2[:,1].max())
                

                data["image_{}".format(idx)]={"area0": area0 ,"point0": point0 ,"count0":0,"area1":  area1,"point1":  point1 ,"count1":0,"area2": area2  ,"point2": point2,"count2":0}                        
                # print(data) 

                ####################################可视化 pridect
                # # 克隆图像
                resultImg =predict[idx,:,:].cpu().numpy().copy()*255
                resultImg= np.uint8(np.clip(resultImg, 0, 255))  # 限制范围在 [0, 255] 之间

                # 创建一个彩色图像
                m_resultImg = cv2.cvtColor(resultImg, cv2.COLOR_GRAY2BGR)

                # 遍历每个连通域点并绘制红色圆圈
                for point in point0:
                    point=tuple(point.tolist())
                    point_change=(point[1],point[0])
                    cv2.circle(m_resultImg, point_change, 1, (0, 0, 255), 15)  # 红色圆圈
                for point in point1:
                    point=tuple(point.tolist())
                    point_change=(point[1],point[0])
                    cv2.circle(m_resultImg, point_change, 1, (0, 255, 0), 15)  # 红色圆圈
                for point in point2:
                    point=tuple(point.tolist())
                    point_change=(point[1],point[0])
                    cv2.circle(m_resultImg, point_change, 1, (255, 255, 0), 15)  # 红色圆圈

                
                # 使用matplotlib显示图像
                # matplotlib默认是RGB格式，所以要将BGR格式转换为RGB
                m_resultImg_rgb = cv2.cvtColor(m_resultImg, cv2.COLOR_BGR2RGB)

                # 显示图像
                plt.figure(figsize=(4,4))
                plt.imshow(m_resultImg_rgb)
                # plt.axis('off')  # 不显示坐标轴
                # plt.show()
                plt.savefig("./active_learning_data/{}_{}/{}/pick/split_{}/picture{}_pridect_points.png".format(seed,otherchoice,"EntropySampling",n,idx))
                plt.close()
                
            else:
                    data["image_{}".format(idx)]={"area0":[] ,"point0": [] ,"count0":0,"area1":  [],"point1":  [] ,"count1":0,"area2": []  ,"point2": [],"count2":0}                        
               
                    continue
        print(data)     
        # with open('/home/user/data/liuyue/active_learning_data/{}_{}/{}/pick/split_{}/data.txt'.format(seed,otherchoice,"EntropySampling",n), 'w') as f:
        #     json.dump(data, f)
        # ########################计算区域内标注的和
        area_sum=torch.zeros([len(idxs),3])
        for i in range(len(idxs)):
            for j in range(3):
                if flag[i]==1:
                    left=data["image_{}".format(i)]["area{}".format(j)][0]
                    right=data["image_{}".format(i)]["area{}".format(j)][1]
                    sum=torch.sum(predict[i,:,left:right])
                    area_sum[i,j]=sum
                    data["image_{}".format(i)]["count{}".format(j)]=sum
        #############################找到和最大的1个区域
        flattened_tensor = area_sum.flatten()
        # 2. 获取最大的 1 个元素的索引
        values, indices = torch.topk(flattened_tensor, 1, largest=True)
        # 3. 将一维索引转换为二维坐标
        # 使用 divmod 来获取行和列
        rows, cols = np.divmod(indices, area_sum.size(1))  # tensor.size(1) 是列数
        # 输出最大的 1个元素的坐标
        coordinates = torch.stack((rows, cols), dim=1)  #each_count中的位置坐标
        print(coordinates )     #a=tensor([[0, 0]])
        # int(a[0][1]) =0        
        #############################找到点数最多的1个区域
        # each_count=torch.zeros([len(idxs),3])
        # for i in range(len(idxs)):
        #     for j in range(3):
        #             each_count[i,j]=data["image_{}".format(i)]["count{}".format(j)]
        # flattened_tensor = each_count.flatten()
        # # 2. 获取最大的 1 个元素的索引
        # values, indices = torch.topk(flattened_tensor, 1, largest=True)
        # # 3. 将一维索引转换为二维坐标
        # # 使用 divmod 来获取行和列
        # rows, cols = np.divmod(indices, each_count.size(1))  # tensor.size(1) 是列数
        # # 输出最大的 3个元素的坐标
        # coordinates = torch.stack((rows, cols), dim=1)  #each_count中的位置坐标
        # print(coordinates )     #a=tensor([[0, 0]])
        # # int(a[0][1]) =0
        ###############################找到点数最多的区域  -》标注
        num_image_pick=int(coordinates[0][0])
        num_area_pick=int(coordinates[0][1])
        flag[ num_image_pick]=0
        left=data["image_{}".format( num_image_pick)]["area{}".format(num_area_pick)][0]   
        right=data["image_{}".format( num_image_pick)]["area{}".format(num_area_pick)][1]   
        print(left,right)  #397,514
        img_area=x.squeeze(1)[num_image_pick][:,left:right]
        mask_area=y.squeeze(1)[num_image_pick][:,left:right]
    ###################################对选定的区域进行可视化
    # # 克隆图像
        resultImg =predict[num_image_pick,:,:].cpu().numpy().copy()*255
        resultImg= np.uint8(np.clip(resultImg, 0, 255))  # 限制范围在 [0, 255] 之间

        # 创建一个彩色图像
        m_resultImg = cv2.cvtColor(resultImg, cv2.COLOR_GRAY2BGR)

        #绘制所选中区域
        a=np.where( predict[num_image_pick,:,left:right]!=1)
        b=(a[1]+int(left),a[0])
        for i in range(b[0].size):
            d=(b[0][i],b[1][i])
            cv2.circle(m_resultImg, d, 1, (160,160,160), 1)

        # 绘制三个区域点
        for point in data["image_{}".format( num_image_pick)]["point0"]:
            point=tuple(point.tolist())
            point_change=(point[1],point[0])
            cv2.circle(m_resultImg, point_change, 1, (0, 0, 255), 15)  # 红色圆圈
        for point in data["image_{}".format( num_image_pick)]["point1"]:
            point=tuple(point.tolist())
            point_change=(point[1],point[0])
            cv2.circle(m_resultImg, point_change, 1, (0, 255, 0), 15)  # 绿色圆圈
        for point in data["image_{}".format( num_image_pick)]["point2"]:
            point=tuple(point.tolist())
            point_change=(point[1],point[0])
            cv2.circle(m_resultImg, point_change, 1, (255, 255, 0), 15)  # 黄色圆圈

        
        # 使用matplotlib显示图像
        # matplotlib默认是RGB格式，所以要将BGR格式转换为RGB
        m_resultImg_rgb = cv2.cvtColor(m_resultImg, cv2.COLOR_BGR2RGB)

        # 显示图像
        plt.figure(figsize=(4,4))
        plt.imshow(m_resultImg_rgb)
        # plt.axis('off')  # 不显示坐标轴
        # plt.show()
        plt.savefig("./active_learning_data/{}_{}/{}/pick/split_{}/picture{}_pickarea_pridect_points.png".format(seed,otherchoice,"EntropySampling",n,num_image_pick))
        plt.close()

        ###############################求这个区域的连通性
        con_nums=[]
        mask_area=mask_area.cpu().numpy().astype(np.uint8)
        # 连通性分析
        num_labels, labels = cv2.connectedComponents(mask_area, connectivity=8)
        output_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

        # 为每个连通组件指定不同的颜色
        for label in range(1, num_labels):  # 0 是背景，跳过
            con_nums.append(label)
            output_image[labels == label] = np.random.randint(0, 255, 3)

      

        # 显示图像
        plt.imshow(output_image)
        plt.axis('off')  # 不显示坐标轴
        plt.savefig("./active_learning_data/{}_{}/{}/pick/split_{}/area{}_connect.png".format(seed,otherchoice,"EntropySampling",n,num_area_pick))
        
        np.savetxt('./active_learning_data/{}_{}/{}/pick/split_{}/area{}_labels.txt'.format(seed,otherchoice,"EntropySampling",n,num_area_pick), labels, fmt='%d', delimiter=',')
        ######labels是一个矩阵，由0，1，2，3，，，，，类
        #########################根据连通性切割小图，
        new_train_imgs_small=torch.zeros([100,224,224])
        new_train_masks_small=torch.zeros([100,224,224])
        id=0
        #################按照断层连通性
        for label in range(1, num_labels):
             patches_img,patches_mask = crop_patches_iterative(img_area,mask_area, label, W, H, max_rounds=3)
            # fids,sids=np.where(labels==label)
            # # for ids in range(len(fids)):
            # #     maskContour.append((sids[ids],fids[ids]))
            # l=abs(fids[-1]-fids[0])
            # w=abs(sids[-1]-sids[0])
            # pickimg=np.zeros([l,w])
            # pickmask=np.zeros([l,w])
            # pickimg=img_area[min(fids[0],fids[-1]):max(fids[0],fids[-1]),min(sids[0],sids[-1]):max(sids[0],sids[-1])]
            # pickmask=mask_area[min(fids[0],fids[-1]):max(fids[0],fids[-1]),min(sids[0],sids[-1]):max(sids[0],sids[-1])]
            # plt.figure(figsize=(4,4))
            # plt.imshow(pickimg.cpu())
            
            # plt.savefig("./active_learning_data/{}_{}/{}/pick/split_{}/{}_img.png".format(seed,otherchoice,"EntropySampling",n,label))


            # plt.figure(figsize=(4,4))
            # plt.imshow(pickmask)
        
            # plt.savefig("./active_learning_data/{}_{}/{}/pick/split_{}/{}_mask.png".format(seed,otherchoice,"EntropySampling",n,label))

           
            
        #     a=fids[0]
        #     b=sids[0]
            
        #     if (sids[-1]-sids[0] )>0:
        #         while(1):
        #             # print(a,b)
        #             if a>288 or b>(right-left-224):
        #                 break
        #             new_train_imgs_small[id]=torch.tensor(img_area[a:a+224,b:b+224])
        #             new_train_masks_small[id]=torch.tensor(mask_area[a:a+224,b:b+224])
                
        #             c=np.where(fids==a+56)
        #             if c[0].size==0:
        #                 break
        #             else:
        #                 a=fids[c[0][0]]
        #                 b=sids[c[0][0]]
        #                 id+=1
        #                 # print(index)
                   

        #     else:
        #         while(1):
        #             print(a,b)
        #             if a>288 or b<224:
        #                 break
        #             new_train_imgs_small[id]=torch.tensor(img_area[a:a+224,b-224:b])
        #             new_train_masks_small[id]=torch.tensor(mask_area[a:a+224,b-224:b])
                    
        #             c=np.where(fids==a+56)
        #             if c[0].size==0:
        #                 break
        #             else:
        #                 a=fids[c[0][0]]
        #                 b=sids[c[0][0]]
        #                 id+=1
        # print(id)
        
        #################按照patch
        # for j in range(4):
            # for k in range(int(right-left)//128):
            #     img1=img_area[j*128:(j+1)*128,k*128:(k+1)*128]
            #     mask1=mask_area[j*128:(j+1)*128,k*128:(k+1)*128]
            #     if mask1.sum()!=0:
            #         new_train_imgs_small[id]=torch.tensor(img1)
            #         new_train_masks_small[id]=torch.tensor(mask1)
            #         id+=1
        # print(id)
        

        new_train_imgs_small=new_train_imgs_small[:id]
        new_train_masks_small=new_train_masks_small[:id]
        if n==1:
                new_train_imgs= new_train_imgs_small
                new_train_masks=new_train_masks_small
                print("new_train_imgs:{}".format(new_train_imgs.shape))
                print("new_train_masks:{}".format(new_train_masks.shape))
        else:
                imgbefor=np.load("./data/liuyue/active_learning/data/THEBE_224/train_img_small.npy")
                maskbefor=np.load("./data/liuyue/active_learning/data/THEBE_224/train_mask_small.npy")
                print("imgbefor:{}".format(imgbefor.shape))
                print("maskbefor:{}".format(maskbefor.shape))
                imgbefor=torch.tensor(imgbefor)
                maskbefor=torch.tensor(maskbefor)
                new_train_imgs_small=torch.tensor(new_train_imgs_small)
                new_train_masks_small=torch.tensor(new_train_masks_small)
                new_train_imgs= torch.cat((new_train_imgs_small, imgbefor), dim=0)
                new_train_masks=torch.cat((new_train_masks_small, maskbefor), dim=0)
                print("new_train_imgs:{}".format(new_train_imgs.shape))
                print("new_train_masks:{}".format(new_train_masks.shape))
        
        
    
        np.save("./data/liuyue/active_learning/data/THEBE_224/train_img_small.npy",new_train_imgs)
        np.save("./data/liuyue/active_learning/data/THEBE_224/train_mask_small.npy",new_train_masks)              
        print(flag)
        return data,flag

        

##########################################
    def predict_prob_LeastConfidence(self, data,n,seed,otherchoice,picknum,picknum_no,flag):
        # self.clf = self.net().to(self.device)
        vit_name="R50-ViT-B_16"
        img_size=224
        vit_patches_size=16
        config_vit = CONFIGS[vit_name]
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (
            int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        self.clf=VisionTransformer(config_vit).to(self.device)
        model_nestunet_path = "./data/liuyue/active_learning_data/{}_{}/{}/SSL_checkpoint_best.pkl".format(seed,otherchoice,"LeastConfidence")
        weights = torch.load(model_nestunet_path, map_location="cuda")['model_state_dict']
        weights_dict = {}
        for k, v in weights.items():
                new_k = k.replace('module.', '') if 'module' in k else k
                weights_dict[new_k] = v
        self.clf.load_state_dict(weights_dict)

        self.clf.eval()
        # prob=[]
        
        
        loader = DataLoader(data, shuffle=False, **self.params['trainsmall_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)

                outputs=np.zeros([len(idxs),2,512,2048])
        for idx in idxs:
            recover_Y_test_pred=predict_slice(THEBE_Net, x[idx].squeeze().cpu(),"LeastConfidence",seed,otherchoice)#512,2048,1
            outputs[idx,1,:,:]=np.squeeze(recover_Y_test_pred)

        outputs[:,0,:,:]=1-outputs[:,1,:,:]
        outputs=torch.tensor(outputs)
        predict= torch.argmax(outputs,dim=1)   
        num=1-torch.max(outputs[:,0,:,:],outputs[:,1,:,:])
        num=num.cpu()


        data={}
        for idx in idxs:
            if flag[idx]==1:
                points=max_50(num[idx])  #50个坐标
                # print(points)
                labels, centroids=kmeans(points)   #labels=0,1,2
                ####################################可视化 聚类的点
                plt.figure(figsize=(4,4))
                plt.scatter(points[:, 1], points[:, 0],s=10, c=labels, cmap='viridis')
                plt.scatter(centroids[:, 1], centroids[:, 0], s=20, c='red', marker='X')  # 绘制簇中心
                plt.title("K-means Clustering (K=3)")
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.savefig("./data/liuyue/active_learning_data/{}_{}/{}/pick/split_{}/picture{}_kmeans.png".format(seed,otherchoice,"LeastConfidence",n,idx))
                plt.close()

               
                #############################创建字典
                labels0=np.where(labels==0)   #索引值
                labels1=np.where(labels==1)
                labels2=np.where(labels==2)
                point0=points[labels0]  #对应的区域点
                point1=points[labels1]
                point2=points[labels2]
                area0=(point0[:,1].min(),point0[:,1].max())
                area1=(point1[:,1].min(),point1[:,1].max())
                area2=(point2[:,1].min(),point2[:,1].max())
                count0=len(labels0[0])
                count1=len(labels1[0])
                count2=len(labels2[0])

                data["image_{}".format(idx)]={"area0": area0 ,"point0": point0 ,"count0":0,"area1":  area1,"point1":  point1 ,"count1":0,"area2": area2  ,"point2": point2,"count2":0}                        
                # print(data) 

                ####################################可视化 pridect
                # # 克隆图像
                resultImg =predict[idx,:,:].cpu().numpy().copy()*255
                resultImg= np.uint8(np.clip(resultImg, 0, 255))  # 限制范围在 [0, 255] 之间

                # 创建一个彩色图像
                m_resultImg = cv2.cvtColor(resultImg, cv2.COLOR_GRAY2BGR)

                # 遍历每个连通域点并绘制红色圆圈
                for point in point0:
                    point=tuple(point.tolist())
                    point_change=(point[1],point[0])
                    cv2.circle(m_resultImg, point_change, 1, (0, 0, 255), 15)  # 红色圆圈
                for point in point1:
                    point=tuple(point.tolist())
                    point_change=(point[1],point[0])
                    cv2.circle(m_resultImg, point_change, 1, (0, 255, 0), 15)  # 红色圆圈
                for point in point2:
                    point=tuple(point.tolist())
                    point_change=(point[1],point[0])
                    cv2.circle(m_resultImg, point_change, 1, (255, 255, 0), 15)  # 红色圆圈

                
                # 使用matplotlib显示图像
                # matplotlib默认是RGB格式，所以要将BGR格式转换为RGB
                m_resultImg_rgb = cv2.cvtColor(m_resultImg, cv2.COLOR_BGR2RGB)

                # 显示图像
                plt.figure(figsize=(4,4))
                plt.imshow(m_resultImg_rgb)
                # plt.axis('off')  # 不显示坐标轴
                # plt.show()
                plt.savefig("./data/liuyue/active_learning_data/{}_{}/{}/pick/split_{}/picture{}_pridect_points.png".format(seed,otherchoice,"LeastConfidence",n,idx))
                plt.close()
                
            else:
                    data["image_{}".format(idx)]={"area0":[] ,"point0": [] ,"count0":0,"area1":  [],"point1":  [] ,"count1":0,"area2": []  ,"point2": [],"count2":0}                        
               
                    continue
        # print(data)     
        # with open('/home/user/data/liuyue/active_learning_data/{}_{}/{}/pick/split_{}/data.txt'.format(seed,otherchoice,"LeastConfidence",n), 'w') as f:
        #     json.dump(data, f)
        # ########################计算区域内标注的和
        area_sum=torch.zeros([len(idxs),3])
        for i in range(len(idxs)):
            for j in range(3):
                if flag[i]==1:
                    left=data["image_{}".format(i)]["area{}".format(j)][0]
                    right=data["image_{}".format(i)]["area{}".format(j)][1]
                    sum=torch.sum(predict[i,:,left:right])
                    area_sum[i,j]=sum
                    data["image_{}".format(i)]["count{}".format(j)]=sum


        #############################找到和最大的1个区域
        flattened_tensor = area_sum.flatten()
        # 2. 获取最大的 1 个元素的索引
        values, indices = torch.topk(flattened_tensor, 1, largest=True)
        # 3. 将一维索引转换为二维坐标
        # 使用 divmod 来获取行和列
        rows, cols = np.divmod(indices, area_sum.size(1))  # tensor.size(1) 是列数
        # 输出最大的 1个元素的坐标
        coordinates = torch.stack((rows, cols), dim=1)  #each_count中的位置坐标
        print(coordinates )     #a=tensor([[0, 0]])
        # int(a[0][1]) =0
        #############################找到点数最多的1个区域
        # each_count=torch.zeros([len(idxs),3])
        # for i in range(len(idxs)):
        #     for j in range(3):
        #             each_count[i,j]=data["image_{}".format(i)]["count{}".format(j)]
        # flattened_tensor = each_count.flatten()
        # # 2. 获取最大的 1 个元素的索引
        # values, indices = torch.topk(flattened_tensor, 1, largest=True)
        # # 3. 将一维索引转换为二维坐标
        # # 使用 divmod 来获取行和列
        # rows, cols = np.divmod(indices, each_count.size(1))  # tensor.size(1) 是列数
        # # 输出最大的 3个元素的坐标
        # coordinates = torch.stack((rows, cols), dim=1)  #each_count中的位置坐标
        # print(coordinates )     #a=tensor([[0, 0]])
        # # int(a[0][1]) =0
        ###############################找到点数最多的区域  -》标注
        num_image_pick=int(coordinates[0][0])
        num_area_pick=int(coordinates[0][1])
        flag[ num_image_pick]=0
        left=data["image_{}".format( num_image_pick)]["area{}".format(num_area_pick)][0]   
        right=data["image_{}".format( num_image_pick)]["area{}".format(num_area_pick)][1]   
        print(left,right)  #397,514
        img_area=x.squeeze(1)[num_image_pick][:,left:right]
        mask_area=y.squeeze(1)[num_image_pick][:,left:right]
    ###################################对选定的区域进行可视化
    # # 克隆图像
        resultImg =predict[num_image_pick,:,:].cpu().numpy().copy()*255
        resultImg= np.uint8(np.clip(resultImg, 0, 255))  # 限制范围在 [0, 255] 之间

        # 创建一个彩色图像
        m_resultImg = cv2.cvtColor(resultImg, cv2.COLOR_GRAY2BGR)

        #绘制所选中区域
        a=np.where( predict[num_image_pick,:,left:right]!=1)
        b=(a[1]+int(left),a[0])
        for i in range(b[0].size):
            d=(b[0][i],b[1][i])
            cv2.circle(m_resultImg, d, 1, (160,160,160), 1)

        # 绘制三个区域点
        for point in data["image_{}".format( num_image_pick)]["point0"]:
            point=tuple(point.tolist())
            point_change=(point[1],point[0])
            cv2.circle(m_resultImg, point_change, 1, (0, 0, 255), 15)  # 红色圆圈
        for point in data["image_{}".format( num_image_pick)]["point1"]:
            point=tuple(point.tolist())
            point_change=(point[1],point[0])
            cv2.circle(m_resultImg, point_change, 1, (0, 255, 0), 15)  # 绿色圆圈
        for point in data["image_{}".format( num_image_pick)]["point2"]:
            point=tuple(point.tolist())
            point_change=(point[1],point[0])
            cv2.circle(m_resultImg, point_change, 1, (255, 255, 0), 15)  # 黄色圆圈

        
        # 使用matplotlib显示图像
        # matplotlib默认是RGB格式，所以要将BGR格式转换为RGB
        m_resultImg_rgb = cv2.cvtColor(m_resultImg, cv2.COLOR_BGR2RGB)

        # 显示图像
        plt.figure(figsize=(4,4))
        plt.imshow(m_resultImg_rgb)
        # plt.axis('off')  # 不显示坐标轴
        # plt.show()
        plt.savefig("./data/liuyue/active_learning_data/{}_{}/{}/pick/split_{}/picture{}_pickarea_pridect_points.png".format(seed,otherchoice,"LeastConfidence",n,num_image_pick))
        plt.close()

        ###############################求这个区域的连通性
        con_nums=[]
        mask_area=mask_area.cpu().numpy().astype(np.uint8)
        # 连通性分析
        num_labels, labels = cv2.connectedComponents(mask_area, connectivity=8)
        output_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

        # 为每个连通组件指定不同的颜色
        for label in range(1, num_labels):  # 0 是背景，跳过
            con_nums.append(label)
            output_image[labels == label] = np.random.randint(0, 255, 3)

        # 显示图像
        plt.imshow(output_image)
        plt.axis('off')  # 不显示坐标轴
        plt.savefig("./data/liuyue/active_learning_data/{}_{}/{}/pick/split_{}/area{}_connect.png".format(seed,otherchoice,"LeastConfidence",n,num_area_pick))
        
        np.savetxt('./data/liuyue/active_learning_data/{}_{}/{}/pick/split_{}/area{}_labels.txt'.format(seed,otherchoice,"LeastConfidence",n,num_area_pick), labels, fmt='%d', delimiter=',')
        ######labels是一个矩阵，由0，1，2，3，，，，，类
        #########################根据连通性切割小图，
        new_train_imgs_small=torch.zeros([100,224,224])
        new_train_masks_small=torch.zeros([100,224,224])
        id=0
        #################按照断层连通性
        for label in range(1, num_labels):
            fids,sids=np.where(labels==label)
            # for ids in range(len(fids)):
            #     maskContour.append((sids[ids],fids[ids]))
            l=abs(fids[-1]-fids[0])
            w=abs(sids[-1]-sids[0])
            pickimg=np.zeros([l,w])
            pickmask=np.zeros([l,w])
            pickimg=img_area[min(fids[0],fids[-1]):max(fids[0],fids[-1]),min(sids[0],sids[-1]):max(sids[0],sids[-1])]
            pickmask=mask_area[min(fids[0],fids[-1]):max(fids[0],fids[-1]),min(sids[0],sids[-1]):max(sids[0],sids[-1])]
            plt.figure(figsize=(4,4))
            plt.imshow(pickimg.cpu())
            
            plt.savefig("./data/liuyue/active_learning_data/{}_{}/{}/pick/split_{}/{}_img.png".format(seed,otherchoice,"LeastConfidence",n,label))


            plt.figure(figsize=(4,4))
            plt.imshow(pickmask)
        
            plt.savefig("./data/liuyue/active_learning_data/{}_{}/{}/pick/split_{}/{}_mask.png".format(seed,otherchoice,"LeastConfidence",n,label))
        
            
            a=fids[0]
            b=sids[0]
            
            if (sids[-1]-sids[0] )>0:
                while(1):
                    # print(a,b)
                    if a>288 or b>(right-left-224):
                        break
                    new_train_imgs_small[id]=torch.tensor(img_area[a:a+224,b:b+224])
                    new_train_masks_small[id]=torch.tensor(mask_area[a:a+224,b:b+224])
                
                    c=np.where(fids==a+56)
                    if c[0].size==0:
                        break
                    else:
                        a=fids[c[0][0]]
                        b=sids[c[0][0]]
                        id+=1
                        # print(index)
                   

            else:
                while(1):
                    print(a,b)
                    if a>288 or b<224:
                        break
                    new_train_imgs_small[id]=torch.tensor(img_area[a:a+224,b-224:b])
                    new_train_masks_small[id]=torch.tensor(mask_area[a:a+224,b-224:b])
                    
                    c=np.where(fids==a+56)
                    if c[0].size==0:
                        break
                    else:
                        a=fids[c[0][0]]
                        b=sids[c[0][0]]
                        id+=1
        # print(id)
        
        #################按照patch
        # for j in range(4):
            # for k in range(int(right-left)//128):
            #     img1=img_area[j*128:(j+1)*128,k*128:(k+1)*128]
            #     mask1=mask_area[j*128:(j+1)*128,k*128:(k+1)*128]
            #     if mask1.sum()!=0:
            #         new_train_imgs_small[id]=torch.tensor(img1)
            #         new_train_masks_small[id]=torch.tensor(mask1)
            #         id+=1
        # print(id)
        

        new_train_imgs_small=new_train_imgs_small[:id]
        new_train_masks_small=new_train_masks_small[:id]
        if n==1:
                new_train_imgs= new_train_imgs_small
                new_train_masks=new_train_masks_small
                print("new_train_imgs:{}".format(new_train_imgs.shape))
                print("new_train_masks:{}".format(new_train_masks.shape))
        else:
                imgbefor=np.load("./data/liuyue/active_learning/data/THEBE_224/train_img_small.npy")
                maskbefor=np.load("./data/liuyue/active_learning/data/THEBE_224/train_mask_small.npy")
                print("imgbefor:{}".format(imgbefor.shape))
                print("maskbefor:{}".format(maskbefor.shape))
                imgbefor=torch.tensor(imgbefor)
                maskbefor=torch.tensor(maskbefor)
                new_train_imgs_small=torch.tensor(new_train_imgs_small)
                new_train_masks_small=torch.tensor(new_train_masks_small)
                new_train_imgs= torch.cat((new_train_imgs_small, imgbefor), dim=0)
                new_train_masks=torch.cat((new_train_masks_small, maskbefor), dim=0)
                print("new_train_imgs:{}".format(new_train_imgs.shape))
                print("new_train_masks:{}".format(new_train_masks.shape))
        
        
    
        np.save("./data/liuyue/active_learning/data/THEBE_224/train_img_small.npy",new_train_imgs)
        np.save("./data/liuyue/active_learning/data/THEBE_224/train_mask_small.npy",new_train_masks)              
        print(flag)
        return data,flag
        

    def predict_prob_BALD_dropout(self, data, n_drop, n, seed,otherchoice,picknum,picknum_no):
            self.clf = self.net().to(self.device)
            model_nestunet_path = "./data/liuyue/active_learning_data/{}_{}/{}/SSL_checkpoint_best.pkl".format(seed,otherchoice,"BALDDropout")
            weights = torch.load(model_nestunet_path, map_location="cuda")['model_state_dict']
            weights_dict = {}
            for k, v in weights.items():
                    new_k = k.replace('module.', '') if 'module' in k else k
                    weights_dict[new_k] = v
            self.clf.load_state_dict(weights_dict)

            
            self.clf.train()
            probs = torch.zeros([picknum,7])  
            probs_no=[]
            num= torch.zeros([n_drop,2,512,2048])
            loader = DataLoader(data, shuffle=False, **self.params['trainsmall_args'])
            maskContour=[]
            maskcount=[]
            maskcount1=[]
            for nd in range(n_drop):
                with torch.no_grad():
                    for x, y, idxs in loader:
                        x, y = x.to(self.device), y.to(self.device)
                        x_select=x[n-1].unsqueeze(0) #1,512,2048
                        
                        y_selsct=y[n-1].unsqueeze(0)  #1,1,512,2048
                        mask0=y.squeeze(1).cpu()
                        out  = self.clf(x_select)
                        
                        outputs=torch.softmax(out, dim=1)
                        out1=outputs[0,1,:,:].detach().cpu().numpy()
                        pred_resnetunet_vision = cv2.applyColorMap((out1 * 255).astype(np.uint8),cmapy.cmap('jet_r'))
                        plt.imshow(pred_resnetunet_vision)
                        plt.savefig("./data/liuyue/active_learning_data/{}_{}/{}/pick/split_{}/nd{}_out.png".format(seed,otherchoice,"BALDDropout",n,nd))
                        
                        
                        predict= torch.argmax(outputs,dim=1)             
                        img1=predict[0,:,:].cpu()
                    
                        plt.figure(figsize=(4,4))
                        plt.imshow(img1,"gray")
                        plt.savefig("./data/liuyue/active_learning_data/{}_{}/{}/pick/split_{}/nd{}_predict.png".format(seed,otherchoice,"BALDDropout",n,nd))
                    
                        
                        mask=y_selsct.squeeze(1)[0,:,:].cpu()
                        plt.figure(figsize=(4,4))
                        plt.imshow(mask,"gray")
                        plt.savefig("./data/liuyue/active_learning_data/{}_{}/{}/pick/split_{}/nd{}_mask.png".format(seed,otherchoice,"BALDDropout",n,nd))
                
                        
                        num[nd,0]=outputs[0,0,:,:]
                        num[nd,1]=outputs[0,1,:,:]

            pb = num.mean(0)
            entropy1 = (-pb*torch.log(pb)).sum(1)
            entropy2 = (-num*torch.log(num)).sum(2).mean(0)
            entr_sum = entropy2 - entropy1
            entr_sum=entr_sum.cpu()
            np.savetxt('./data/liuyue/active_learning_data/{}_{}/{}/pick/split_{}/entr_sum.txt'.format(seed,otherchoice,"BALDDropout",n), entr_sum, fmt='%d', delimiter=',')
            id=0
            while(1):
                num_min=entr_sum.min()
                num_max=entr_sum.max()
                fid=np.where(entr_sum==num_max)[0][0]
                # print(fid)
                
                sid=np.where(entr_sum==num_max)[1][0]
                if predict[0,fid,sid]==1:
                    probs[id][0]=fid
                    probs[id][1]=sid
                    maskContour.append((sid,fid))
                    probs[id][2]=num_max
                    probs[id][3]=outputs[0,0,fid,sid]
                    probs[id][4]=outputs[0,1,fid,sid]
                    probs[id][5]=predict[0,fid,sid]
                    probs[id][6]=mask0[0,fid,sid]
                    id+=1
                
                else:
                    probs_no.append((fid,sid))
                    maskContour.append((sid,fid))
                entr_sum[fid][sid]=num_min-1

                if id==picknum:
                    break
            if len(probs_no)  <picknum_no:
                while(1):
                    num_min=entr_sum.min()
                    num_max=entr_sum.max()
                    fid=np.where(entr_sum==num_max)[0][0]
                    # print(fid)
                    
                    sid=np.where(entr_sum==num_max)[1][0]
                    probs_no.append((fid,sid)) 
                    maskcount1.append((sid,fid))
                    entr_sum[fid][sid]=num_min-1
                    if len(probs_no)  ==picknum_no:
                            break
                        

            
            # 克隆图像
            resultImg = mask.numpy().copy()*255
            resultImg= np.uint8(np.clip(resultImg, 0, 255))  # 限制范围在 [0, 255] 之间

            # 创建一个彩色图像
            m_resultImg = cv2.cvtColor(resultImg, cv2.COLOR_GRAY2BGR)

            # 遍历每个连通域点并绘制红色圆圈
            for point in maskContour:
                cv2.circle(m_resultImg, point, 1, (0, 0, 255), 10)  # 红色圆圈

            for point in maskcount1:
                cv2.circle(m_resultImg, point, 1, (0, 255, 225), 10)

            # 使用matplotlib显示图像
            # matplotlib默认是RGB格式，所以要将BGR格式转换为RGB
            m_resultImg_rgb = cv2.cvtColor(m_resultImg, cv2.COLOR_BGR2RGB)

            # 显示图像
            plt.imshow(m_resultImg_rgb)
            # plt.axis('off')  # 不显示坐标轴
            # plt.show()
            plt.savefig("./data/liuyue/active_learning_data/{}_{}/{}/pick/split_{}/_mask_points.png".format(seed,otherchoice,"BALDDropout",n))



            new_train_imgs_small=np.zeros([picknum+picknum_no,128,128])
            new_train_masks_small=np.zeros([picknum+picknum_no,128,128])
            pick_train_imgs=np.zeros([picknum,128,128])
            pick_train_masks=np.zeros([picknum,128,128])
            image=x_select.squeeze().cpu()
            masks=y_selsct.squeeze().cpu()     
            bigimage=np.zeros([640,2176])
            bigmask=np.zeros([640,2176])
            for i in range(64,576):
                for j in range(64,2112):
                    bigimage[i][j]=image[i-64][j-64]
                    bigmask[i][j]=masks[i-64][j-64]
            for i in range(picknum):
                firstid=probs[i][0]+64
                secondid=probs[i][1]+64
                for j in range(128):
                    for z in range(128):
                        pick_train_imgs[i][j][z]=bigimage[int(firstid-64+j)][int(secondid-64+z)]
                        pick_train_masks[i][j][z]=bigmask[int(firstid-64+j)][int(secondid-64+z)]
                pick_train_masks[i][64][64]=5
                plt.figure(figsize=(4,4))
                plt.imshow(pick_train_imgs[i])
                plt.savefig("./data/liuyue/active_learning_data/{}_{}/{}/pick/split_{}/{}_img.png".format(seed,otherchoice,"BALDDropout",n,i))

                plt.figure(figsize=(4,4))
                plt.imshow(pick_train_masks[i])
                plt.savefig("./data/liuyue/active_learning_data/{}_{}/{}/pick/split_{}/{}_mask.png".format(seed,otherchoice,"BALDDropout",n,i))

            for i in range(picknum):
                firstid=probs[i][0]
                secondid=probs[i][1]
                if firstid<64:
                    firstid=64
                if firstid>448:
                    firstid=448
                if secondid<64:
                    secondid=64
                if secondid>1984:
                    secondid=1984
                for j in range(128):
                    for z in range(128):
                        new_train_imgs_small[i][j][z]=image[int(firstid-64+j)][int(secondid-64+z)]
                        new_train_masks_small[i][j][z]=masks[int(firstid-64+j)][int(secondid-64+z)]
            
                for i in range(picknum_no):
                    firstid,secondid=probs_no[i]
                    
                    if firstid<64:
                        firstid=64
                    if firstid>448:
                        firstid=448
                    if secondid<64:
                        secondid=64
                    if secondid>1984:
                        secondid=1984
                    for j in range(128):
                        for z in range(128):
                            new_train_imgs_small[i+picknum][j][z]=image[int(firstid-64+j)][int(secondid-64+z)]
                            new_train_masks_small[i+picknum][j][z]=masks[int(firstid-64+j)][int(secondid-64+z)]


            if n==1:
                    new_train_imgs= new_train_imgs_small
                    new_train_masks=new_train_masks_small
                    print("new_train_imgs:{}".format(new_train_imgs.shape))
                    print("new_train_masks:{}".format(new_train_masks.shape))
            else:
                    imgbefor=np.load("./data/liuyue/active_learning/data/THEBE_NEW/train_img_small.npy")
                    maskbefor=np.load("./data/liuyue/active_learning/data/THEBE_NEW/train_mask_small.npy")
                    print("imgbefor:{}".format(imgbefor.shape))
                    print("maskbefor:{}".format(maskbefor.shape))
                    imgbefor=torch.tensor(imgbefor)
                    maskbefor=torch.tensor(maskbefor)
                    new_train_imgs_small=torch.tensor(new_train_imgs_small)
                    new_train_masks_small=torch.tensor(new_train_masks_small)
                    new_train_imgs= torch.cat((new_train_imgs_small, imgbefor), dim=0)
                    new_train_masks=torch.cat((new_train_masks_small, maskbefor), dim=0)
                    print("new_train_imgs:{}".format(new_train_imgs.shape))
                    print("new_train_masks:{}".format(new_train_masks.shape))
            
            
        
            np.save("./data/liuyue/active_learning/data/THEBE_NEW/train_img_small.npy",new_train_imgs)
            np.save("./data/liuyue/active_learning/data/THEBE_NEW/train_mask_small.npy",new_train_masks)
                        
                        
            return probs
        
   


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor,smooth):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape

    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0
    iou = (intersection + smooth) / (union + smooth)  # We smooth our devision to avoid 0/0
    return iou

class faultsDataset(torch.utils.data.Dataset):
    def __init__(self,preprocessed_images):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.images = preprocessed_images
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = TF.to_tensor(image)
        image=norm(image)
        image = TF.normalize(image, [4.0902375e-05, ], [0.0383472, ])
        return image



class BCEDiceLoss(nn.Module):
    def __init__(self, **kwargs):
        super(BCEDiceLoss, self).__init__()
        self.bce_func =  nn.BCELoss()
        # self.dice_func = BinaryDiceLoss()#使用binarydiceloss
        # self.dice_func=soft_cldice_loss()#使用soft_cldice_loss

    # loss = loss_f(outputs_1.cpu(), outputs.cpu(), labels.cpu())
    def forward(self, predict, target):
        loss_bce=self.bce_func(predict,target)
        # loss_dice=self.dice_func(predict,target)
        # return 0.5*loss_dice + 0.5*loss_bce
        return loss_bce
    

import torch
import torch.nn.functional as F
from torch import nn





def min_50(tensor):
        #  # 创建一个随机的 tensor 矩阵（假设为二维矩阵）
        # tensor = torch.randn(100, 100)  # 例如，一个 10x10 的矩阵

        # 1. 将 Tensor 展开为一维
        flattened_tensor = tensor.flatten()

        # 2. 获取最小的50 个元素的索引
        values, indices = torch.topk(flattened_tensor, 50, largest=False)

        # 3. 将一维索引转换为二维坐标
        # 使用 divmod 来获取行和列
        rows, cols = np.divmod(indices, tensor.size(1))  # tensor.size(1) 是列数

        # 输出最小的 50 个元素的坐标
        coordinates = torch.stack((rows, cols), dim=1)
        # print("最小的 100 个像素点的坐标：", coordinates)
        return coordinates

def max_50(tensor):
        #  # 创建一个随机的 tensor 矩阵（假设为二维矩阵）
        # tensor = torch.randn(100, 100)  # 例如，一个 10x10 的矩阵

        # 1. 将 Tensor 展开为一维
        flattened_tensor = tensor.flatten()

        # 2. 获取最小的 100 个元素的索引
        values, indices = torch.topk(flattened_tensor, 50, largest=True)

        # 3. 将一维索引转换为二维坐标
        # 使用 divmod 来获取行和列
        rows, cols = np.divmod(indices, tensor.size(1))  # tensor.size(1) 是列数

        # 输出最大的 50 个元素的坐标
        coordinates = torch.stack((rows, cols), dim=1)
        # print("最小的 100 个像素点的坐标：", coordinates)
        return coordinates




def kmeans(coordinates ):
    

        # # 假设这50个坐标点是如下的随机数据
        # coordinates = np.random.rand(50, 2)  # 50个二维坐标点，数据范围是[0,1]

        # 使用 K-means 聚类
        kmeans = KMeans(n_clusters=3, random_state=42)  # 设置为3类
        kmeans.fit(coordinates)

        # 获取每个点的分类标签
        labels = kmeans.labels_

        # 获取簇中心
        centroids = kmeans.cluster_centers_
        return labels, centroids
       




class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

class SVHN_Net(nn.Module):
    def __init__(self):
        super(SVHN_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1152, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 1152)
        x = F.relu(self.fc1(x))
        e1 = F.relu(self.fc2(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc3(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

class CIFAR10_Net(nn.Module):
    def __init__(self):
        super(CIFAR10_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50
    

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class FAULTSEG_Net(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,###############################################
                 bilinear: bool = True,
                 base_c: int = 32):
        super(FAULTSEG_Net, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.size()[1] == 1 and self.in_channels == 3:  # 如果channel 是1，变成3
            x = x.repeat(1, 3, 1, 1)
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        logits=torch.sigmoid(logits)
        return logits



class PowerAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride, p=2):
        super(PowerAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.p = p  # The power to which to raise each element before averaging

    def forward(self, x):
        # Apply power (raise to the power of p) to the input
        x = torch.pow(x, self.p)  # Raise each element to the power of p
        # Apply average pooling
        x = nn.functional.avg_pool2d(x, self.kernel_size, self.stride)
        # Apply inverse power to return to the original scale
        x = torch.pow(x, 1 / self.p)  # Inverse power to recover from the raised value
        return x

# self.pool = PowerAvgPool2d(kernel_size=2, stride=2, p=2)  # Using power of 2

class THEBE_Net(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,###############################################
                 bilinear: bool = True,
                 base_c: int = 64):
        super(THEBE_Net, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.size()[1] == 1 and self.in_channels == 3:  # 如果channel 是1，变成3
            x = x.repeat(1, 3, 1, 1)
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        # logits=torch.sigmoid(logits)
        return logits







class ConvBlock(nn.Module):
    """Basic convolutional block with two 3x3 convolutions and ReLU activations."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

def crop_patches_iterative(img,mask, coords, W, H, max_rounds=3):
    coords = np.asarray(coords).copy()
    patches_img = []
    patches_mask = []

    for _ in range(max_rounds):
        if len(coords) == 0:
            break

        rows = coords[:, 0]
        cols = coords[:, 1]

        left_idx = np.argmin(cols)
        right_idx = np.argmax(cols)
        left_point = coords[left_idx]
        right_point = coords[right_idx]

        if left_point[0] < right_point[0]:
            cur = left_point
        else:
            cur = right_point

        y, x = int(cur[0]), int(cur[1])

        while True:
            removed = False

            if 0 <= y and y + H <= img.shape[0] and 0 <= x and x + W <= img.shape[1]:
                patchi = img[y:y+H, x:x+W].copy()
                patches_img.append(patch)
                patchm = mask[y:y+H, x:x+W].copy()
                patches_mask.append(patch)

                in_patch = (
                    (coords[:, 0] >= y) & (coords[:, 0] < y + H) &
                    (coords[:, 1] >= x) & (coords[:, 1] < x + W)
                )
                if np.any(in_patch):
                    coords = coords[~in_patch]
                    removed = True
                else:
                    removed = False

            if not removed:
                break

            # 重新建立 row_to_first
            rows = coords[:, 0]
            cols = coords[:, 1]
            order = np.lexsort((cols, rows))
            coords_sorted = coords[order]
            row_to_first = {}
            for r, c in coords_sorted:
                if r not in row_to_first:
                    row_to_first[r] = (r, c)

            # 更新 y'
            y_next = int(round(y + W / 2))
            if y_next == y:
                y_next = y + 1

            if y_next not in row_to_first:
                break

            y, x = row_to_first[y_next]

    return patches_img,patches_mask
