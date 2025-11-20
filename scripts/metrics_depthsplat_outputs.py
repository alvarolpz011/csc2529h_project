import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from skimage import io
import torch


OUTPUT_PATH= "/u/alvarolopez/Documents/csc2529/csc2529h_project/depthsplat/outputs/dl3dv_single_scene"
OUTPUT_IMAGES_PATH= OUTPUT_PATH + "/images"

ORIGINAL_DATASET_PATH= "/w/20251/alvarolopez/datasets"

AVAILABLE_RESULTS_SCENES_HASHES= os.listdir(OUTPUT_IMAGES_PATH)

SCENES_IMAGES_PATHS= [OUTPUT_IMAGES_PATH + "/" + scene_hash + "/color" for scene_hash in AVAILABLE_RESULTS_SCENES_HASHES]

import torch.nn.functional as F

def rescale_images
def compute_psnr_resizable(gt, pred, target_size=None):
    """PSNR with automatic resizing."""
    print(gt.shape)
    print(pred.shape)
    if gt.shape != pred.shape:
        print("resizing")
        # Resize pred to match gt
        pred = F.interpolate(
            pred.unsqueeze(0) if pred.ndim == 3 else pred,
            size=gt.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        if pred.ndim == 4:
            pred = pred.squeeze(0)
    
    # Standard PSNR calculation
    mse = ((gt - pred) ** 2).mean()
    psnr = -10 * (mse.log10())
    return psnr

for scene in AVAILABLE_RESULTS_SCENES_HASHES:
    psnrs=[]
    output_images = sorted(os.listdir(OUTPUT_IMAGES_PATH + "/" + scene + "/color"))
    ground_truth_all= sorted(os.listdir(ORIGINAL_DATASET_PATH + "/" + scene + "/nerfstudio/images_8"))
    subset_ground_truth= ground_truth_all[:len(output_images)]
    
    for i in range(len(output_images)):
        output_image_path= OUTPUT_IMAGES_PATH + "/" + scene + "/color/" + output_images[i]
        ground_truth_image_path= ORIGINAL_DATASET_PATH + "/" + scene + "/nerfstudio/images_8/" + subset_ground_truth[i]
        
        print(output_image_path)
        print(ground_truth_image_path)
        
        output_image= io.imread(output_image_path) / 255.0
        print(output_image.shape)
        ground_truth_image= io.imread(ground_truth_image_path) / 255.0
        print(ground_truth_image.shape)
        
        #computing psnr
        psnr_value= compute_psnr_resizable(
            torch.tensor(ground_truth_image).permute(2,0,1), 
            torch.tensor(output_image).permute(2,0,1)
        ).item()
        
        psnrs.append(psnr_value)
        
        #ssim_value= ssim(ground_truth_image, output_image, multichannel=True, data_range=1.0)
        
        print(f"Scene: {scene}, Image: {output_images[i]}, PSNR: {psnr_value}, SSIM: {np.nan}")
    print(f"Mean PSNR: {np.mean(psnrs)}")