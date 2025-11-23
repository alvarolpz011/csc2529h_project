import torch
import os
import json
import argparse
import numpy as np
import random
import cv2
from PIL import Image, ImageFilter
from tqdm import tqdm
from torchvision import transforms

# Metrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance

from model import Difix

def load_model_weights(model, ckpt_path):
    print(f"Loading weights from {ckpt_path}...")
    sd = torch.load(ckpt_path, map_location="cpu")
    
    if model.adapter and "state_dict_adapter" in sd:
        model.adapter.load_state_dict(sd["state_dict_adapter"])
    if model.adapter_canny and "state_dict_adapter_canny" in sd:
        model.adapter_canny.load_state_dict(sd["state_dict_adapter_canny"])
        
    if "state_dict_vae" in sd:
        model.vae.load_state_dict(sd["state_dict_vae"], strict=False)
        
    model.eval()
    return model

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

def get_canny_edge(image):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)

def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Initialize Metrics
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device) 
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    
    if os.path.isdir(args.dataset_path):
        dataset_name = os.path.basename(os.path.normpath(args.dataset_path))
        args.dataset_path = os.path.join(args.dataset_path, f"{dataset_name}.json")

    # Load Data
    with open(args.dataset_path, 'r') as f:
        data = json.load(f)
        test_data = data['test']
    test_ids = sorted(list(test_data.keys()))

    # Transforms
    T_img = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
    ])
    T_norm = transforms.Normalize([0.5], [0.5])
    T_cond = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    print(f"\n--- Evaluating Model: {args.run_name} ---")
    model = Difix(
        timestep=199, 
        use_depth_adapter=args.use_depth,
        use_canny_adapter=args.use_canny
    ).to(device)
    model = load_model_weights(model, args.checkpoint_path)

    results = {"psnr": [], "ssim": [], "lpips": []}

    for img_id in tqdm(test_ids):
        item = test_data[img_id]
        
        # Load Images
        gt_pil = Image.open(item['image']).convert("RGB")
        input_pil = gt_pil.copy()

        # Apply Degradation
        if args.degrade_inputs:
            random.seed(int(img_id) if img_id.isdigit() else hash(img_id)) 
            input_pil = simulate_artifacts(input_pil)

        # Prepare Tensors
        gt_tensor = T_img(gt_pil).unsqueeze(0).to(device)       # [1, 3, H, W], range [0, 1]
        input_tensor = T_img(input_pil).unsqueeze(0).to(device) # [1, 3, H, W], range [0, 1]
        model_input = T_norm(input_tensor)                      # [1, 3, H, W], range [-1, 1]

        # Prepare Conditions
        depth_map = None
        if args.use_depth:
            depth_pil = Image.open(item['depth_image']).convert("L")
            depth_map = T_cond(depth_pil).unsqueeze(0).unsqueeze(0).to(device)

        canny_map = None
        if args.use_canny:
            if 'canny_image' in item:
                canny_pil = Image.open(item['canny_image']).convert("L")
            else:
                canny_pil = get_canny_edge(input_pil).convert("L") 
            canny_map = T_cond(canny_pil).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            out_tensor = model(
                model_input.unsqueeze(0),
                prompt=item.get('prompt', args.prompt), 
                depth_map=depth_map,
                canny_map=canny_map
            )
            
            # Post-process: [-1, 1] -> [0, 1]
            pred_tensor = (out_tensor[:, 0] * 0.5 + 0.5).clamp(0, 1)

            # --- Update Standard Metrics ---
            results["psnr"].append(psnr(pred_tensor, gt_tensor).item())
            results["ssim"].append(ssim(pred_tensor, gt_tensor).item())
            results["lpips"].append(lpips(pred_tensor, gt_tensor).item())

            # --- Update FID (Requires uint8 [0, 255]) ---
            gt_uint8 = (gt_tensor * 255).to(dtype=torch.uint8)
            pred_uint8 = (pred_tensor * 255).to(dtype=torch.uint8)
            
            fid.update(gt_uint8, real=True)
            fid.update(pred_uint8, real=False)

    # Final Stats
    avg_psnr = np.mean(results["psnr"])
    avg_ssim = np.mean(results["ssim"])
    avg_lpips = np.mean(results["lpips"])
    
    # Compute FID over the whole set
    print("Computing FID... (this might take a moment)")
    fid_score = fid.compute().item()

    print("="*65)
    print(f"RESULTS FOR: {args.run_name}")
    print(f"PSNR:  {avg_psnr:.4f}")
    print(f"SSIM:  {avg_ssim:.4f}")
    print(f"LPIPS: {avg_lpips:.4f}")
    print(f"FID:   {fid_score:.4f}")
    print("="*65)
    
    eval_dir = os.path.dirname(args.checkpoint_path).replace("checkpoints", "eval")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
        json.dump({
            "psnr": avg_psnr, 
            "ssim": avg_ssim, 
            "lpips": avg_lpips,
            "fid": fid_score,
            "args": vars(args)
        }, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--run_name", type=str, default="experiment")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--prompt", type=str, default="high quality 3d render")
    
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--use_canny", action="store_true")
    parser.add_argument("--degrade_inputs", action="store_true")
    
    args = parser.parse_args()
    evaluate(args)