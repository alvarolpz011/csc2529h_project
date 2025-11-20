# csc2529h_project

[ ] Write about exactly the difference between 3dgs and NeRF


[ ] (Shayan's) Look into other 3dgs methods that in some way incorporate the D channel (same kind of dataset  and choose that as a baseline)

[ ] (Shayan's) Find specific dataset (or even two)
* DL3DV
* RealEstate10K
* Tartan Air
* ScanNet

### Notes
* The code for Difix3D is in [this github](https://github.com/nv-tlabs/Difix3D).
* A paper did something similar using depth estimation models: [DepthSplat: Connecting Gaussian Splatting and Depth](https://arxiv.org/pdf/2410.13862). We could check what datasets they've used.

### Potential problems: 
* What if we don't find a dataset that includes Depth information?
* Do we have access to the finetuned model they made for DIFIX or the dataset they used? If we dont have the model they used, we would need to replicate their finetuning, which requires the data they used.
* Could we use depth estimation models instead of Lidar points?
* I can't find the exact Stable Difussion model they used as base, just the newest version in Stability AI's Hugging Face page.

## SETTING UP ENVIRONMENT
To make envoronment that can run both DepthSplat and Difix3D (recommended to use Conda):
1. Install Python 3.10.14 or create new conda env with that version. The name will be `depthsplat_difix` for these examples.
    ```bash
    conda create -n depthsplat_difix python=3.10.14
    ```
2. Activate the env:
    ```bash
    conda activate depthsplat_difix
    ```
3. Following Oleksii's instructions for Depthsplat environment (because we need to install the diff-gaussian-rasterzation package manually):
    1. `pip install --upgrade pip wheel`
    2. Go to the diff-gaussian-rasterization-modified submodule dir: `cd diff-gaussian-rasterization-modified`
    3. `nano cuda_rasterizer/rasterizer_impl.h`
    4. **At the very top** (near other `#include` lines), add:
        ```cpp
        #include <cstdint>
        ```
    5. Initialize submodules: `git submodule update --init --recursive`
    6. Install PyTorch with CUDA 12.4 support: 
        ```bash
        pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
        ```

    7. Install the package: 
        ```bash
        pip install --no-build-isolation ../diff-gaussian-rasterization-modified
        ```
    8. Go to DepthSplat submodule dir: 
        ```bash
        cd ../depthsplat/
        ```
    9. Remove the original `diff-gaussian-rasterization` line from `requirements.txt` (we installed the fixed version manually):
        ```bash
        sed -i '/diff-gaussian-rasterization/d' requirements.txt
        ```
    10. `pip install -r requirements.txt`
4. `cd ../Difix3D`
5. `pip install -r requirements.txt`
6. `pip install pandas`
7. Al done. You should have an environment that can run both DepthSplat and Diffix code.

## Setting up the DL3DV Dataset:

The `download_dl3dv.py` script will be in the `scripts` folder. You have to add your Hugging Face token inside download.py to authenticate.

Note that the path where the dataset is downloaded will be important for running the loop.

### Example Commands

```bash
# Download scene with hash 0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695
python download_dl3dv.py  --subset hash --hash 0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695 --odir /w/20251/<your_user_dir>/datasets --only_level8

# Download scene with hash 9641a1ed7963ce5ca734cff3e6ccea3dfa8bcb0b0a3ff78f65d32a080de2d71e
python download.py  --subset hash --hash 9641a1ed7963ce5ca734cff3e6ccea3dfa8bcb0b0a3ff78f65d32a080de2d71e --odir /w/20251/<your_user_dir>/datasets --only_level8
```

> **Notes:**
> * Each scene may take ~40 minutes to download at 10 MB/s.
> * You must add your Hugging Face token inside download.py to authenticate.
> * Full list of scene hashes can be found [here](https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark/blob/main/benchmark-meta.csv).



## Running DepthSplat + Difix
With the dataset downloaded dataset, now all that's left to do is run the loop with `run_loop.sh`. The code is set to run 5 steps in the DepthSplat-Difix loop. 

Before running `run_loop.sh`, you'll need to set up the variables at the top of the file:

* `DEPTHSPLAT_OUTPUTS_BASE`: just the directory where you want the depthsplat outputs to be saved.
* `DEPTHSPLAT_DIR`= the directory where the main depthsplat directory is.
* `ORIGINAL_DOWNLOADED_DATASET_DIR`= the directory where you downloaded the dl3dv datasets.
* `DIFIX_OUTPUTS_BASE`= the directory where you want the outputs of the difix pipeline to be saved.


To run, you just need to execute:

```bash
bash run_loop.sh
```