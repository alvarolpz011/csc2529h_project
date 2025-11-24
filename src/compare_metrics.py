import os
import json
import glob
import pandas as pd

# 1. Configuration
ROOT_DIR = "outputs"

# 2. Find all metrics files
# Pattern matches: outputs / {dataset} / {mode} / {variant} / eval / metrics.json
search_pattern = os.path.join(ROOT_DIR, "*", "*", "*", "eval", "metrics.json")
files = glob.glob(search_pattern)

print(f"Found {len(files)} metric files.")

data_records = []

# 3. Parse files
for file_path in files:
    try:
        rel_path = os.path.relpath(file_path, ROOT_DIR)
        parts = rel_path.split(os.sep)
        
        if len(parts) < 5:
            print(f"Skipping path: {rel_path}")
            continue

        dataset = parts[0]
        mode = parts[1]
        variant = parts[2]

        with open(file_path, 'r') as f:
            metrics = json.load(f)

        record = {
            "dataset": dataset,
            "mode": mode,
            "variant": variant,
            "psnr": metrics.get("psnr", 0),
            "ssim": metrics.get("ssim", 0),
            "lpips": metrics.get("lpips", 0),
            "fid": metrics.get("fid", 0)
        }
        data_records.append(record)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# 4. Aggregate with Pandas
if not data_records:
    print("No data found.")
else:
    df = pd.DataFrame(data_records)

    summary = df.groupby(['mode', 'variant'])[['psnr', 'ssim', 'lpips', 'fid']].mean().reset_index()

    # Sort for better readability (e.g., clean first, then degraded)
    summary = summary.sort_values(by=['mode', 'variant'])

    print("\n=== Aggregated Results (Mean across all datasets) ===")
    print(summary.to_markdown(index=False, floatfmt=".4f"))

    # summary.to_csv("aggregated_results.csv", index=False)