import torch
import os
import json
import argparse
import numpy as np
import random
from PIL import Image, ImageFilter # Added ImageFilter
from tqdm import tqdm
from torchvision import transforms
import torch.nn.functional as F

# Metrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from model import Difix

def load_model_weights(model, ckpt_path, use_depth):
    print(f"Loading weights from {ckpt_path}...")
    sd = torch.load(ckpt_path, map_location="cpu")
    
    if use_depth and "state_dict_adapter" in sd:
        model.adapter.load_state_dict(sd["state_dict_adapter"])
        
    if "state_dict_vae" in sd:
        model.vae.load_state_dict(sd["state_dict_vae"], strict=False)
        
    model.eval()
    return model

# NEW: Copy of the degradation logic
def simulate_artifacts(img_pil):
    if random.random() < 0.8:
        radius = random.uniform(1, 3)
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius))
    if random.random() < 0.5:
        w, h = img_pil.size
        factor = random.uniform(2, 4)
        small_w, small_h = int(w/factor), int(h/factor)
        img_pil = img_pil.resize((small_w, small_h), resample=Image.BILINEAR)
        img_pil = img_pil.resize((w, h), resample=Image.NEAREST)
    return img_pil

def evaluate(args):
    device = "cuda"
    
    # Metrics setup...
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

    with open(args.dataset_path, 'r') as f:
        data = json.load(f)
        test_data = data['test']
    test_ids = sorted(list(test_data.keys()))

    T_img = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
    ])
    T_norm = transforms.Normalize([0.5], [0.5])
    
    T_depth = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    def run_inference(ckpt_path, use_depth_adapter, description):
        print(f"\n--- Evaluating: {description} ---")
        model = Difix(timestep=199, use_depth_adapter=use_depth_adapter).to(device)
        load_model_weights(model, ckpt_path, use_depth_adapter)
        
        results = {"psnr": [], "ssim": [], "lpips": []}
        
        for img_id in tqdm(test_ids):
            item = test_data[img_id]
            
            # Load Images
            gt_pil = Image.open(item['image']).convert("RGB")
            input_pil = gt_pil.copy() # Start with clean

            # NEW: Apply degradation if flag is set
            if args.degrade_inputs:
                # Use fixed seed per image for consistency between model runs
                random.seed(img_id) 
                input_pil = simulate_artifacts(input_pil)

            # Ground Truth (Clean)
            gt_tensor = T_img(gt_pil).unsqueeze(0).unsqueeze(0).to(device) 
            
            # Input (Potentially Degraded)
            input_tensor = T_img(input_pil).unsqueeze(0).unsqueeze(0).to(device)
            model_input = T_norm(input_tensor)
            
            depth_map = None
            if use_depth_adapter:
                depth_pil = Image.open(item['depth_image']).convert("L")
                depth_map = T_depth(depth_pil).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                out_tensor = model(model_input, prompt=item['prompt'], depth_map=depth_map)
                pred_tensor = (out_tensor[:, 0] * 0.5 + 0.5).clamp(0, 1)
                gt_metric_tensor = gt_tensor[:, 0]

            results["psnr"].append(psnr(pred_tensor, gt_metric_tensor).item())
            results["ssim"].append(ssim(pred_tensor, gt_metric_tensor).item())
            results["lpips"].append(lpips(pred_tensor, gt_metric_tensor).item())

        avg_psnr = np.mean(results["psnr"])
        avg_ssim = np.mean(results["ssim"])
        avg_lpips = np.mean(results["lpips"])
        print(f"{description} -> PSNR: {avg_psnr:.4f} | LPIPS: {avg_lpips:.4f}")
        
        del model
        torch.cuda.empty_cache()
        return avg_psnr, avg_ssim, avg_lpips

    s_psnr, s_ssim, s_lpips = run_inference(args.ckpt_standard, False, "Standard Difix")
    d_psnr, d_ssim, d_lpips = run_inference(args.ckpt_depth, True, "Depth Difix")

    print("\n" + "="*65)
    print(f"{'Metric':<10} | {'Standard':<12} | {'Depth-Guided':<12} | {'Delta':<10}")
    print("-" * 65)
    print(f"{'PSNR':<10} | {s_psnr:<12.4f} | {d_psnr:<12.4f} | {d_psnr - s_psnr:+.4f}")
    print(f"{'SSIM':<10} | {s_ssim:<12.4f} | {d_ssim:<12.4f} | {d_ssim - s_ssim:+.4f}")
    print(f"{'LPIPS':<10} | {s_lpips:<12.4f} | {d_lpips:<12.4f} | {d_lpips - s_lpips:+.4f}")
    print("="*65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/lego.json")
    parser.add_argument("--ckpt_standard", type=str, required=True)
    parser.add_argument("--ckpt_depth", type=str, required=True)
    parser.add_argument("--degrade_inputs", action="store_true", help="Corrupt inputs during evaluation")
    args = parser.parse_args()

    evaluate(args)