import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
from tqdm import tqdm

def compute_psnr(gt, pred):
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return 100.0
    return -10 * np.log10(mse)

def find_images_dir(base_path):
    """
    Recursively find the directory containing .png files.
    Prioritizes paths ending in 'color' or 'renders' inside a hash-like folder.
    """
    base_path = Path(base_path)
    
    if len(list(base_path.glob("*.png"))) > 0:
        return base_path

    for item in base_path.iterdir():
        if item.is_dir():
            potential_color = item / "color"
            potential_renders = item / "renders"
            
            if potential_color.exists() and len(list(potential_color.glob("*.png"))) > 0:
                return potential_color
            if potential_renders.exists() and len(list(potential_renders.glob("*.png"))) > 0:
                return potential_renders
            
            if len(list(item.glob("*.png"))) > 0:
                return item

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True, help="Path to current step output images")
    parser.add_argument("--gt_root", type=str, required=True, help="Root of original downloaded dataset")
    args = parser.parse_args()

    pred_base = Path(args.pred_dir)
    gt_root = Path(args.gt_root)

    pred_dir = find_images_dir(pred_base)
    if pred_dir is None:
        print(f"[Error] No PNG images found in {pred_base} or its subdirectories.")
        return

    scene_hash = None
    for item in gt_root.iterdir():
        if item.is_dir() and len(item.name) > 20:
            scene_hash = item.name
            break
    
    if scene_hash is None:
        print("[Error] Could not find a hash folder in GT root.")
        return

    gt_dir = gt_root / scene_hash / "nerfstudio/images_8"
    
    print("="*40)
    print(f"[EVAL] Configuration")
    print(f"  Pred Dir: {pred_dir}")
    print(f"  GT Dir:   {gt_dir}")
    print("="*40)

    pred_files = sorted([f for f in pred_dir.glob("*.png")])
    gt_files = sorted([f for f in gt_dir.glob("*.png")])

    if len(pred_files) == 0:
        print("[Error] Prediction directory is empty.")
        return

    psnr_accum = 0.0
    ssim_accum = 0.0
    count = 0
    
    limit = min(len(pred_files), len(gt_files))
    
    print(f"[EVAL] Processing {limit} frames...")

    for i in tqdm(range(limit)):
        pred_img = io.imread(pred_files[i]).astype(np.float32) / 255.0
        gt_img = io.imread(gt_files[i]).astype(np.float32) / 255.0

        pred_tensor = torch.from_numpy(pred_img).permute(2, 0, 1).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0)

        if pred_tensor.shape != gt_tensor.shape:
            pred_tensor = F.interpolate(
                pred_tensor, 
                size=(gt_tensor.shape[2], gt_tensor.shape[3]), 
                mode='bilinear', 
                align_corners=False
            )
            pred_img = pred_tensor.squeeze(0).permute(1, 2, 0).numpy()

        psnr_val = compute_psnr(gt_img, pred_img)
        
        ssim_val = ssim(gt_img, pred_img, channel_axis=2, data_range=1.0, win_size=3)

        psnr_accum += psnr_val
        ssim_accum += ssim_val
        count += 1

    avg_psnr = psnr_accum / count
    avg_ssim = ssim_accum / count

    print("\n" + "="*40)
    print(f"TRUE METRICS (vs Original GT)")
    print(f"PSNR: {avg_psnr:.4f} dB")
    print(f"SSIM: {avg_ssim:.4f}")
    print("="*40 + "\n")

    metrics_path = pred_base.parent / "true_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"PSNR: {avg_psnr}\nSSIM: {avg_ssim}\n")
        print(f"[Info] Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()