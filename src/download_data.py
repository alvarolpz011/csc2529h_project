import os
import json
import argparse
import random
from pathlib import Path
from huggingface_hub import snapshot_download

PROMPTS = {
    "lego": "a high quality lego bulldozer",
    "drums": "a red drum set on a checkered floor",
}

def create_json(data_root, dataset_name):
    data_root = Path(data_root)
    output_path = data_root / f"{dataset_name}.json"
    
    source_dir = data_root / "test"
    
    if not source_dir.exists():
        print(f"  [Warning] Source directory {source_dir} not found. Skipping JSON creation.")
        return

    all_files = sorted(list(source_dir.glob("r_*.png")))
    rgb_files = [f for f in all_files if "depth" not in f.name]

    if not rgb_files:
        print("  [Warning] No RGB files found matching pattern r_*.png.")
        return

    data_entries = []
    prompt = PROMPTS.get(dataset_name, "a high quality 3d render")

    print(f"  Processing {len(rgb_files)} images...")

    for rgb_path in rgb_files:
        file_id = rgb_path.stem
        
        depth_name = f"{file_id}_depth_0001.png"
        depth_path = source_dir / depth_name
        
        entry = {
            "id": file_id,
            "image": str(rgb_path.absolute()),
            "target_image": str(rgb_path.absolute()),
            "prompt": prompt
        }

        if depth_path.exists():
            entry["depth_image"] = str(depth_path.absolute())
        else:
            print(f"  [Info] Missing depth for {file_id}")
            pass

        data_entries.append(entry)

    random.seed(42)
    random.shuffle(data_entries)
    
    split_idx = int(len(data_entries) * 0.9)
    train_data = {item.pop("id"): item for item in data_entries[:split_idx]}
    test_data = {item.pop("id"): item for item in data_entries[split_idx:]}

    json_data = {
        "train": train_data,
        "test": test_data
    }

    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)
        
    print(f"  JSON saved to: {output_path}")
    print(f"  Train: {len(train_data)} | Test: {len(test_data)}")

def download_and_process(dataset_name, root_dir):
    print(f"[{dataset_name.upper()}] Starting Sequence...")
    
    # 1. Download
    print("  Downloading from Hugging Face...")
    try:
        local_dir = snapshot_download(
            repo_id="rishitdagli/nerf-gs-datasets",
            repo_type="dataset",
            allow_patterns=f"{dataset_name}/*",
            local_dir=root_dir,
            local_dir_use_symlinks=False,
            tqdm_class=None
        )
    except Exception as e:
        print(f"  Error downloading: {e}")
        return

    target_dir = Path(root_dir) / dataset_name
    print(f"  Data located at: {target_dir}")

    # 2. Create JSON
    print("  Generating dataset JSON...")
    create_json(target_dir, dataset_name)
    print(f"[{dataset_name.upper()}] Done.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lego", choices=["lego", "drums"], help="Name of the dataset to download")
    parser.add_argument("--data_dir", type=str, default="data", help="Root directory to store data")
    args = parser.parse_args()
    
    download_and_process(args.dataset, args.data_dir)