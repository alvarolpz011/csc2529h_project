#!/usr/bin/env python3
import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser(description="Trim transforms.json to top-K frames.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Base directory containing hash subfolders.")
    args = parser.parse_args()

    base_dir = args.data_dir

    # Iterate through every subdirectory (each 'hash')
    for hash_name in os.listdir(base_dir):
        hash_path = os.path.join(base_dir, hash_name)
        if not os.path.isdir(hash_path):
            continue  # skip non-directories

        nerfstudio_dir = os.path.join(hash_path, "nerfstudio")
        transforms_path = os.path.join(nerfstudio_dir, "transforms.json")
        images_dir = os.path.join(nerfstudio_dir, "images_8")

        if not os.path.exists(transforms_path):
            print(f"[WARN] No transforms.json found for {hash_name}, skipping.")
            continue

        if not os.path.exists(images_dir):
            print(f"[WARN] No images_8/ directory found for {hash_name}, skipping.")
            continue

        # Count images → K
        K = len([
            f for f in os.listdir(images_dir)
            if os.path.isfile(os.path.join(images_dir, f))
        ])

        print(f"[INFO] {hash_name}: K = {K} images")

        # Load the original JSON
        with open(transforms_path, "r") as f:
            data = json.load(f)

        # Trim frames
        original_count = len(data.get("frames", []))
        data["frames"] = data["frames"][:K]

        print(f"[INFO] {hash_name}: Trimmed frames {original_count} → {len(data['frames'])}")

        # Write back to the same file
        with open(transforms_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"[OK]   Updated {transforms_path}")

if __name__ == "__main__":
    main()
