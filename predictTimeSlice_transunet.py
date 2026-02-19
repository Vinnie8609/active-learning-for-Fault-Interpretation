import os

import torchvision.transforms.functional as TF

from image_tools import *
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


import matplotlib.pyplot as plt
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def predict_slice(model_name,seis,strategy_name,seed,otherchoice):
    Z, XL = seis.shape
    batch_size=8
    im_height = Z
    im_width = XL
    splitsize = 224  # 96
    stepsize = 112  # overlap half
    overlapsize = splitsize - stepsize

    horizontal_splits_number = int(np.ceil((im_width) / stepsize))
    width_after_pad = stepsize * horizontal_splits_number + 2 * overlapsize
    left_pad = int((width_after_pad - im_width) / 2)
    right_pad = width_after_pad - im_width - left_pad

    vertical_splits_number = int(np.ceil((im_height) / stepsize))
    height_after_pad = stepsize * vertical_splits_number + 2 * overlapsize

    top_pad = int((height_after_pad - im_height) / 2)
    bottom_pad = height_after_pad - im_height - top_pad

    horizontal_splits_number = horizontal_splits_number + 1
    vertical_splits_number = vertical_splits_number + 1

    X_list = []

    X_list.extend(
        split_Image(seis, True, top_pad, bottom_pad, left_pad, right_pad, splitsize, stepsize, vertical_splits_number,
                    horizontal_splits_number))

    X = np.asarray(X_list)

    faults_dataset_test = faultsDataset(X)

    test_loader = torch.utils.data.DataLoader(dataset=faults_dataset_test,
                                              batch_size=batch_size,
                                              shuffle=False)
    # 加载模型
    test_predictions = []
    imageNo = -1
    mergemethod = "smooth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vit_name="R50-ViT-B_16"
    img_size=224
    vit_patches_size=16
    config_vit = CONFIGS[vit_name]
    if vit_name.find('R50') != -1:
            config_vit.patches.grid = (
            int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    model=VisionTransformer(config_vit).to(device)

    model_nestunet_path = "./active_learning_data/{}_{}/{}/SSL_checkpoint_best.pkl".format(seed,otherchoice,strategy_name).format(seed,otherchoice,strategy_name)
    
   
    weights = torch.load(model_nestunet_path, map_location="cuda")['model_state_dict']
    weights_dict = {}
    for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
    model.load_state_dict(weights_dict)

    model.eval()
    for images in test_loader:
        images = images.type(torch.FloatTensor)
        images = images.to(device)
        outputs = model(images)
        
        y_preds=outputs.squeeze(1)
        test_predictions.extend(y_preds.detach().cpu())
        # print(y_preds.shape)
        if len(test_predictions) >= vertical_splits_number * horizontal_splits_number:
            imageNo = imageNo + 1
            tosave = torch.stack(test_predictions).detach().cpu().numpy()[
                     0:vertical_splits_number * horizontal_splits_number]
            test_predictions = test_predictions[vertical_splits_number * horizontal_splits_number:]

            if mergemethod == "smooth":
                WINDOW_SPLINE_2D = window_2D(window_size=splitsize, power=2)
                # add one dimension
                tosave = np.expand_dims(tosave, -1)
                tosave = np.array([patch * WINDOW_SPLINE_2D for patch in tosave])  # 224,224,450
                tosave = tosave.reshape((vertical_splits_number, horizontal_splits_number, splitsize, splitsize, 1))
                recover_Y_test_pred = recover_Image(tosave, (im_height, im_width, 1), left_pad, right_pad, top_pad,
                                                    bottom_pad, overlapsize)

    return recover_Y_test_pred
