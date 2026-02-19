
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict
from common_tools import create_logger
import losses
import os
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from image_tools import norm

import torch.utils.data
from sklearn.cluster import KMeans


from predictTimeSlice_transunet import predict_slice

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
        if "R50" in vit_name:
            config_vit.patches.grid = (
                int(img_size / vit_patches_size),
                int(img_size / vit_patches_size),
            )

        self.clf=VisionTransformer(config_vit).to(self.device)


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

        # 定义 Warmup 学习率策�?

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


                img=predicted_mask.squeeze(1)[0,:,:].cpu()
                plt.imshow(img)
                plt.savefig("./active_learning_data/{}_{}/{}/picture/val/{}_{}.png".format(seed,otherchoice,strategy_name,n,int(idxs[0])))

                mask=y.squeeze(1)[0,:,:].cpu()
                plt.imshow(mask)
                plt.savefig("./active_learning_data/{}_{}/{}/picture/val/{}_{}_mask.png".format(seed,otherchoice,strategy_name,n,int(idxs[0])))
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




        optimizer = optim.AdamW(self.clf.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=0.001)

        # 定义 Warmup 学习率策�?

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

                out = self.clf(x)
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


                img=predicted_mask.squeeze(1)[0,:,:].cpu()
                plt.imshow(img)
                plt.savefig("./active_learning_data/{}_{}/{}/picture/val/{}_{}.png".format(seed,otherchoice,strategy_name,n,int(idxs[0])))

                mask=y.squeeze(1)[0,:,:].cpu()
                plt.imshow(mask)
                plt.savefig("./active_learning_data/{}_{}/{}/picture/val/{}_{}_mask.png".format(seed,otherchoice,strategy_name,n,int(idxs[0])))
        return best_miou




    def predict_prob_EntropySampling(self, data, round_idx, seed, otherchoice, flag):
        vit_name = "R50-ViT-B_16"
        img_size = 224
        vit_patches_size = 16
        config_vit = CONFIGS[vit_name]
        if "R50" in vit_name:
            config_vit.patches.grid = (
                int(img_size / vit_patches_size),
                int(img_size / vit_patches_size),
            )
        self.clf = VisionTransformer(config_vit).to(self.device)
        model_path = "./active_learning_data/{}_{}/{}/SSL_checkpoint_best.pkl".format(
            seed, otherchoice, "EntropySampling"
        )
        weights = torch.load(model_path, map_location="cuda")["model_state_dict"]
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace("module.", "") if "module" in k else k
            weights_dict[new_k] = v
        self.clf.load_state_dict(weights_dict)
        self.clf.eval()

        pick_dir = os.path.join(
            "./active_learning_data",
            "{}_{}".format(seed, otherchoice),
            "EntropySampling",
            "pick",
            "split_{}".format(round_idx),
        )
        os.makedirs(pick_dir, exist_ok=True)

        loader = DataLoader(data, shuffle=False, **self.params["trainsmall_args"])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                prob_map = np.zeros([len(idxs), 2, 512, 2048])

        for bi, idx in enumerate(idxs):
            pred_slice = predict_slice(
                THEBE_Net, x[bi].squeeze().cpu(), "EntropySampling", seed, otherchoice
            )
            recover = np.squeeze(pred_slice)
            if recover.shape == (2048, 512):
                recover = recover.T
            prob_map[bi, 1, :, :] = recover

        prob_map[:, 0, :, :] = 1 - prob_map[:, 1, :, :]
        prob_map = torch.tensor(prob_map)
        pred_mask = torch.argmax(prob_map, dim=1)

        prob0 = prob_map[:, 0, :, :]
        prob1 = prob_map[:, 1, :, :]
        log0 = torch.log(prob0)
        log1 = torch.log(prob1)
        log0 = torch.nan_to_num(log0, nan=0.0)
        log1 = torch.nan_to_num(log1, nan=0.0)
        entropy_map = -(prob0 * log0 + prob1 * log1)
        entropy_map = entropy_map.cpu()

        pick_data = {}
        for bi, idx in enumerate(idxs):
            idx_int = int(idx)
            if flag[idx_int] == 1:
                top_points = max_50(entropy_map[bi])
                cluster_labels, cluster_centroids = kmeans(top_points)
                plt.figure(figsize=(4, 4))
                plt.scatter(top_points[:, 1], top_points[:, 0], s=10, c=cluster_labels, cmap="viridis")
                plt.scatter(
                    cluster_centroids[:, 1],
                    cluster_centroids[:, 0],
                    s=20,
                    c="red",
                    marker="X",
                )
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.savefig(os.path.join(pick_dir, "picture{}_kmeans.png".format(idx_int)))
                plt.close()

                labels0 = np.where(cluster_labels == 0)
                labels1 = np.where(cluster_labels == 1)
                labels2 = np.where(cluster_labels == 2)
                points0 = top_points[labels0]
                points1 = top_points[labels1]
                points2 = top_points[labels2]
                area0 = (points0[:, 1].min(), points0[:, 1].max())
                area1 = (points1[:, 1].min(), points1[:, 1].max())
                area2 = (points2[:, 1].min(), points2[:, 1].max())

                pick_data["image_{}".format(idx_int)] = {
                    "area0": area0,
                    "point0": points0,
                    "count0": 0,
                    "area1": area1,
                    "point1": points1,
                    "count1": 0,
                    "area2": area2,
                    "point2": points2,
                    "count2": 0,
                }

                result_img = pred_mask[bi, :, :].cpu().numpy().copy() * 255
                result_img = np.uint8(np.clip(result_img, 0, 255))
                result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)

                for point in points0:
                    point = tuple(point.tolist())
                    point_change = (point[1], point[0])
                    cv2.circle(result_img_bgr, point_change, 1, (0, 0, 255), 15)
                for point in points1:
                    point = tuple(point.tolist())
                    point_change = (point[1], point[0])
                    cv2.circle(result_img_bgr, point_change, 1, (0, 255, 0), 15)
                for point in points2:
                    point = tuple(point.tolist())
                    point_change = (point[1], point[0])
                    cv2.circle(result_img_bgr, point_change, 1, (255, 255, 0), 15)

                result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(4, 4))
                plt.imshow(result_img_rgb)
                plt.savefig(os.path.join(pick_dir, "picture{}_pridect_points.png".format(idx_int)))
                plt.close()
            else:
                pick_data["image_{}".format(idx_int)] = {
                    "area0": [],
                    "point0": [],
                    "count0": 0,
                    "area1": [],
                    "point1": [],
                    "count1": 0,
                    "area2": [],
                    "point2": [],
                    "count2": 0,
                }
                continue

        area_sum = torch.zeros([len(idxs), 3])
        for bi, idx in enumerate(idxs):
            idx_int = int(idx)
            for j in range(3):
                if flag[idx_int] == 1:
                    left = pick_data["image_{}".format(idx_int)]["area{}".format(j)][0]
                    right = pick_data["image_{}".format(idx_int)]["area{}".format(j)][1]
                    total = torch.sum(pred_mask[bi, :, left:right])
                    area_sum[bi, j] = total
                    pick_data["image_{}".format(idx_int)]["count{}".format(j)] = total

        flattened_tensor = area_sum.flatten()
        values, indices = torch.topk(flattened_tensor, 1, largest=True)
        rows, cols = np.divmod(indices, area_sum.size(1))
        coordinates = torch.stack((rows, cols), dim=1)

        picked_image_idx = int(coordinates[0][0])
        picked_area_idx = int(coordinates[0][1])
        picked_image_global_idx = int(idxs[picked_image_idx])
        flag[picked_image_global_idx] = 0
        left = pick_data["image_{}".format(picked_image_global_idx)]["area{}".format(picked_area_idx)][0]
        right = pick_data["image_{}".format(picked_image_global_idx)]["area{}".format(picked_area_idx)][1]

        img_area = x.squeeze(1)[picked_image_idx][:, left:right]
        mask_area = y.squeeze(1)[picked_image_idx][:, left:right]

        result_img = pred_mask[picked_image_idx, :, :].cpu().numpy().copy() * 255
        result_img = np.uint8(np.clip(result_img, 0, 255))
        result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
        selected = np.where(pred_mask[picked_image_idx, :, left:right] != 1)
        selected = (selected[1] + int(left), selected[0])
        for i in range(selected[0].size):
            d = (selected[0][i], selected[1][i])
            cv2.circle(result_img_bgr, d, 1, (160, 160, 160), 1)

        for point in pick_data["image_{}".format(picked_image_global_idx)]["point0"]:
            point = tuple(point.tolist()) if hasattr(point, "tolist") else tuple(point)
            point_change = (point[1], point[0])
            cv2.circle(result_img_bgr, point_change, 1, (0, 0, 255), 15)
        for point in pick_data["image_{}".format(picked_image_global_idx)]["point1"]:
            point = tuple(point.tolist()) if hasattr(point, "tolist") else tuple(point)
            point_change = (point[1], point[0])
            cv2.circle(result_img_bgr, point_change, 1, (0, 255, 0), 15)
        for point in pick_data["image_{}".format(picked_image_global_idx)]["point2"]:
            point = tuple(point.tolist()) if hasattr(point, "tolist") else tuple(point)
            point_change = (point[1], point[0])
            cv2.circle(result_img_bgr, point_change, 1, (255, 255, 0), 15)

        result_img_rgb = cv2.cvtColor(result_img_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(4, 4))
        plt.imshow(result_img_rgb)
        plt.savefig(os.path.join(pick_dir, "picture{}_pickarea_pridect_points.png".format(picked_image_idx)))
        plt.close()

        mask_area = mask_area.cpu().numpy().astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(mask_area, connectivity=8)
        output_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
        for label in range(1, num_labels):
            output_image[labels == label] = np.random.randint(0, 255, 3)

        plt.imshow(output_image)
        plt.axis("off")
        plt.savefig(os.path.join(pick_dir, "area{}_connect.png".format(picked_area_idx)))
        np.savetxt(
            os.path.join(pick_dir, "area{}_labels.txt".format(picked_area_idx)),
            labels,
            fmt="%d",
            delimiter=",",
        )

        max_patches = 100
        patch_height = 224
        patch_width = 224
        new_train_imgs_small = torch.zeros([max_patches, patch_height, patch_width])
        new_train_masks_small = torch.zeros([max_patches, patch_height, patch_width])
        patch_count = 0
        for label in range(1, num_labels):
            coords = np.column_stack(np.where(labels == label))
            if coords.size == 0:
                continue
            patches_img, patches_mask = crop_patches_iterative(
                img_area, mask_area, coords, patch_width, patch_height, max_rounds=3
            )
            if len(patches_img) == 0:
                continue
            new_train_imgs_small[patch_count:patch_count + len(patches_img), :, :] = torch.tensor(patches_img)
            new_train_masks_small[patch_count:patch_count + len(patches_mask), :, :] = torch.tensor(patches_mask)
            patch_count += len(patches_img)

        new_train_imgs_small = new_train_imgs_small[:patch_count]
        new_train_masks_small = new_train_masks_small[:patch_count]
        if round_idx == 1:
            new_train_imgs = new_train_imgs_small
            new_train_masks = new_train_masks_small
        else:
            imgbefor = np.load("./data/liuyue/active_learning/data/THEBE_224/train_img_small.npy")
            maskbefor = np.load("./data/liuyue/active_learning/data/THEBE_224/train_mask_small.npy")
            imgbefor = torch.tensor(imgbefor)
            maskbefor = torch.tensor(maskbefor)
            new_train_imgs_small = torch.tensor(new_train_imgs_small)
            new_train_masks_small = torch.tensor(new_train_masks_small)
            new_train_imgs = torch.cat((new_train_imgs_small, imgbefor), dim=0)
            new_train_masks = torch.cat((new_train_masks_small, maskbefor), dim=0)

        save_dir = "./data/liuyue/active_learning/data/THEBE_224"
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "train_img_small.npy"), new_train_imgs)
        np.save(os.path.join(save_dir, "train_mask_small.npy"), new_train_masks)

        return pick_data, flag

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, smooth):
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
        #使用binarydiceloss
        #使用soft_cldice_loss


    def forward(self, predict, target):
        loss_bce=self.bce_func(predict,target)


        return loss_bce







def min_50(tensor):
        #  # 创建一个随机的 tensor 矩阵
        flattened_tensor = tensor.flatten()
        # 2. 获取最小的50 个元素的索引
        values, indices = torch.topk(flattened_tensor, 50, largest=False)

        # 3. 将一维索引转换为二维坐标
        # 使用 divmod 来获取行和列
        rows, cols = np.divmod(indices, tensor.size(1))  # tensor.size(1) 
        # 输出最小的 50 个元素的坐标
        coordinates = torch.stack((rows, cols), dim=1)
        # print("最小的 100 个像素点的坐标：", coordinates)
        return coordinates

def max_50(tensor):
        #  # 创建一个随机的 tensor 矩阵
        flattened_tensor = tensor.flatten()
        # 2. 获取最小的 100 个元素的索引
        values, indices = torch.topk(flattened_tensor, 50, largest=True)

        # 3. 将一维索引转换为二维坐标
        # 使用 divmod 来获取行和列
        rows, cols = np.divmod(indices, tensor.size(1))  # tensor.size(1) 
        # 输出最大的 50 个元素的坐标
        coordinates = torch.stack((rows, cols), dim=1)
        # print("最小的 100 个像素点的坐标：", coordinates)
        return coordinates




def kmeans(coordinates ):


       
        # 使用 K-means 聚类
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(coordinates)
        # 获取每个点的分类标签
        labels = kmeans.labels_

      
        centroids = kmeans.cluster_centers_
        return labels, centroids








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
        if x.size()[1] == 1 and self.in_channels == 3:  #
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
                patches_img.append(patchi)
                patchm = mask[y:y+H, x:x+W].copy()
                patches_mask.append(patchm)

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
