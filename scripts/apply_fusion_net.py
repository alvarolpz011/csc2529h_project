import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch

from fusion_net import load_fusion_model


def load_img(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return arr


def save_img(arr, path):
    arr = np.clip(arr * 255.0, 0, 255).astype("uint8")
    img = Image.fromarray(np.transpose(arr, (1, 2, 0)))
    img.save(path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply a trained fusion network to combine DepthSplat and DIFFIX outputs.\n"
            "Takes two directory trees (depthsplat, diffix) with matching relative structure "
            "and writes fused images to output_dir."
        )
    )
    parser.add_argument("--depthsplat_dir", type=str, required=True)
    parser.add_argument("--diffix_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True, help="Path to fusion network weights (.pt).")
    args = parser.parse_args()

    ds_root = Path(args.depthsplat_dir).expanduser().resolve()
    fx_root = Path(args.diffix_dir).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_fusion_model(args.weights, map_location=device)
    model.to(device)
    model.eval()

    exts = {".png", ".jpg", ".jpeg"}
    samples = []
    for dirpath, _, filenames in os.walk(ds_root):
        for fname in filenames:
            if Path(fname).suffix.lower() not in exts:
                continue
            d_path = Path(dirpath) / fname
            rel = d_path.relative_to(ds_root)
            f_path = fx_root / rel
            if f_path.exists():
                samples.append((d_path, f_path, rel))

    if not samples:
        print("[apply_fusion_net] No matching pairs found. Check directory structure.")
        return

    print(f"[apply_fusion_net] Found {len(samples)} pairs.")
    print(f"[apply_fusion_net] Device: {device}")

    for d_path, f_path, rel in samples:
        d = load_img(d_path)
        f = load_img(f_path)

        # resize both to the same size (DepthSplat size)
        C, H, W = d.shape

        def resize_arr(arr):
            c, h, w = arr.shape
            img = Image.fromarray((np.transpose(arr, (1, 2, 0)) * 255.0).astype("uint8"))
            img = img.resize((W, H), Image.BILINEAR)
            arr2 = np.asarray(img).astype("float32") / 255.0
            return np.transpose(arr2, (2, 0, 1))

        f = resize_arr(f)

        x = np.concatenate([d, f], axis=0)  # (6, H, W)
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # (1, 6, H, W)

        with torch.no_grad():
            y_t = model(x_t)
        y = y_t.squeeze(0).cpu().numpy()  # (3, H, W)

        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_img(y, out_path)

    print(f"[apply_fusion_net] Done. Fused images saved under {out_root}")


if __name__ == "__main__":
    main()
