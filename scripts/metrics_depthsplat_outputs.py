import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from skimage import io



OUTPUT_PATH= "/u/alvarolopez/Documents/csc2529/depthsplat/depthsplat/outputs/dl3dv_single_scene"
OUTPUT_IMAGES_PATH= OUTPUT_PATH + "/images"

ORIGINAL_DATASET_PATH= "/w/20251/alvarolopez/datasets"

AVAILABLE_RESULTS_SCENES_HASHES= os.listdir(OUTPUT_IMAGES_PATH)

SCENES_IMAGES_PATHS= [OUTPUT_IMAGES_PATH + "/" + scene_hash + "/color" for scene_hash in AVAILABLE_RESULTS_SCENES_HASHES]


for scene in AVAILABLE_RESULTS_SCENES_HASHES:
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
        
        psnr_value= psnr(ground_truth_image, output_image, data_range=1.0)
        ssim_value= ssim(ground_truth_image, output_image, multichannel=True, data_range=1.0)
        
        print(f"Scene: {scene}, Image: {output_images[i]}, PSNR: {psnr_value}, SSIM: {ssim_value}")
    print(output_images)
    print(subset_ground_truth)
