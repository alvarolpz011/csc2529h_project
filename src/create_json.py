import os
import json
import argparse
import random
from pathlib import Path

def create_json(data_root, output_path, prompt="a high quality lego bulldozer"):
    data_root = Path(data_root)
    
    # In this specific dataset structure, ONLY the 'test' folder has depth maps.
    # We will use it as our source and split it manually.
    source_dir = data_root / "test"
    
    if not source_dir.exists():
        print(f"Error: {source_dir} does not exist.")
        return

    # Find all RGB images (start with r_, end with .png, exclude depth files)
    all_files = sorted(list(source_dir.glob("r_*.png")))
    rgb_files = [f for f in all_files if "depth" not in f.name]

    if not rgb_files:
        print("No RGB files found.")
        return

    data_entries = []

    for rgb_path in rgb_files:
        # Extract ID (e.g. "r_0" from "r_0.png")
        file_id = rgb_path.stem
        
        # Construct expected depth filename based on your ls -R output
        # Pattern: r_{id}_depth_0001.png
        depth_name = f"{file_id}_depth_0001.png"
        depth_path = source_dir / depth_name
        
        if not depth_path.exists():
            print(f"Skipping {file_id}: Depth map {depth_name} missing.")
            continue

        entry = {
            "id": file_id,
            "image": str(rgb_path),
            "target_image": str(rgb_path), # Self-supervised reconstruction
            "depth_image": str(depth_path),
            "prompt": prompt
        }
        data_entries.append(entry)

    # Shuffle and Split 90% Train / 10% Test
    random.seed(42)
    random.shuffle(data_entries)
    
    split_idx = int(len(data_entries) * 0.9)
    train_data = {item.pop("id"): item for item in data_entries[:split_idx]}
    test_data = {item.pop("id"): item for item in data_entries[split_idx:]}

    json_data = {
        "train": train_data,
        "test": test_data
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)
        
    print(f"JSON created at {output_path}")
    print(f"Total Pairs Found: {len(data_entries)}")
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/lego", help="Path to lego dataset root")
    parser.add_argument("--output", type=str, default="data/lego.json")
    args = parser.parse_args()
    
    create_json(args.data_dir, args.output)