import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

from fusion_net import SimpleFusionNet


def load_img(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype("float32") / 255.0
    # (H, W, C) -> (C, H, W)
    arr = np.transpose(arr, (2, 0, 1))
    return arr


class FusionDataset(Dataset):
    """
    Dataset of triplets: (DepthSplat, DIFFIX, GT).

    Assumes three directory trees with the same relative layout.
    """

    def __init__(self, depthsplat_root: Path, diffix_root: Path, gt_root: Path):
        self.depthsplat_root = depthsplat_root
        self.diffix_root = diffix_root
        self.gt_root = gt_root

        self.samples = []
        exts = {".png", ".jpg", ".jpeg"}
        for dirpath, _, filenames in os.walk(gt_root):
            for fname in filenames:
                if Path(fname).suffix.lower() not in exts:
                    continue
                gt_path = Path(dirpath) / fname
                rel = gt_path.relative_to(gt_root)
                d_path = depthsplat_root / rel
                f_path = diffix_root / rel
                if d_path.exists() and f_path.exists():
                    self.samples.append((d_path, f_path, gt_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d_path, f_path, g_path = self.samples[idx]
        d = load_img(d_path)
        f = load_img(f_path)
        g = load_img(g_path)
        # resize all to match GT
        _, H, W = g.shape
        # torch expects CHW, but we resize with PIL/np route:
        def resize_arr(arr):
            c, h, w = arr.shape
            img = Image.fromarray((np.transpose(arr, (1, 2, 0)) * 255.0).astype("uint8"))
            img = img.resize((W, H), Image.BILINEAR)
            arr2 = np.asarray(img).astype("float32") / 255.0
            return np.transpose(arr2, (2, 0, 1))

        d = resize_arr(d)
        f = resize_arr(f)

        # concat depth + diffix along channel dim -> (6, H, W)
        x = np.concatenate([d, f], axis=0)
        return (
            torch.from_numpy(x),
            torch.from_numpy(g),
        )


def train_fusion(
    depthsplat_dir: str,
    diffix_dir: str,
    gt_dir: str,
    out_path: str,
    batch_size: int = 4,
    num_epochs: int = 5,
    lr: float = 1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_root = Path(depthsplat_dir).expanduser().resolve()
    fx_root = Path(diffix_dir).expanduser().resolve()
    gt_root = Path(gt_dir).expanduser().resolve()

    dataset = FusionDataset(ds_root, fx_root, gt_root)
    if len(dataset) == 0:
        raise RuntimeError("No matching triplets found for training. Check directory structure.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = SimpleFusionNet().to(device)
    optim_ = optim.Adam(model.parameters(), lr=lr)

    print(f"[train_fusion_net] dataset size: {len(dataset)}")
    print(f"[train_fusion_net] device: {device}")
    print(f"[train_fusion_net] epochs: {num_epochs}, batch_size: {batch_size}, lr: {lr}")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x, g in loader:
            x = x.to(device)
            g = g.to(device)

            optim_.zero_grad()
            y = model(x)
            loss = F.mse_loss(y, g)
            loss.backward()
            optim_.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"[train_fusion_net] epoch {epoch+1}/{num_epochs}, loss = {avg_loss:.6f}")

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"[train_fusion_net] saved weights to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tiny fusion network for DepthSplat + DIFFIX.")
    parser.add_argument("--depthsplat_dir", type=str, required=True, help="Directory with DepthSplat outputs (e.g., step0).")
    parser.add_argument("--diffix_dir", type=str, required=True, help="Directory with DIFFIX outputs corresponding to depthsplat_dir.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory with ground-truth images (resized or will be resized to match).")
    parser.add_argument("--out_path", type=str, required=True, help="Where to save the trained weights (.pt).")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_fusion(
        depthsplat_dir=args.depthsplat_dir,
        diffix_dir=args.diffix_dir,
        gt_dir=args.gt_dir,
        out_path=args.out_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
    )
