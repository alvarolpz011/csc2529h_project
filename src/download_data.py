from huggingface_hub import snapshot_download
import os

# Download specifically the 'lego' folder from the dataset
print("Downloading Lego dataset with depth from Hugging Face...")
local_dir = snapshot_download(
    repo_id="rishitdagli/nerf-gs-datasets",
    repo_type="dataset",
    allow_patterns="lego/*",
    local_dir="data",
    local_dir_use_symlinks=False
)

print(f"Download complete. Data located at: {os.path.join(local_dir, 'lego')}")