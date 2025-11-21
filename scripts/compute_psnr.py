import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
from math import log10

def psnr(gt, pr):
    gt = gt.astype("float32")/255.0
    pr = pr.astype("float32")/255.0
    mse = np.mean((gt-pr)**2)
    if mse == 0:
        return float("inf")
    return 10*log10(1.0/mse)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gt_root = Path(args.gt_dir)
    pr_root = Path(args.pred_dir)
    scores=[]
    for dirpath,_,files in os.walk(gt_root):
        for f in files:
            if f.lower().endswith((".png",".jpg",".jpeg")):
                gt_path = Path(dirpath)/f
                rel = gt_path.relative_to(gt_root)
                pr_path = pr_root/rel
                if not pr_path.exists():
                    continue
                gt_img = np.asarray(Image.open(gt_path).convert("RGB"))
                pr_img = np.asarray(Image.open(pr_path).convert("RGB").resize(gt_img.shape[1::-1]))
                scores.append(psnr(gt_img, pr_img))
    if scores:
        print(f"PSNR[{args.tag}] = {np.mean(scores):.3f} dB")
    else:
        print("No pairs.")

if __name__ == "__main__":
    main()
