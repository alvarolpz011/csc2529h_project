# DepthSplat Setup Instructions on HPC (Python 3.10 + CUDA)

These instructions are tailored for a Linux HPC environment with limited `~` space (10 GB) and large `/w` scratch space.

Make sure you are on the gpunode with CUDA available.
Replace `oleksii` with your own username in paths below.

---

## 1. Install Python 3.10 locally

```bash
mkdir -p ~/local/python310
cd ~/local
wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz
tar -xvf Python-3.10.14.tgz
cd Python-3.10.14
./configure --prefix=$HOME/local/python310 --enable-optimizations
make -j4
make install
```

Create virtual environment on `/w`:

```bash
~/local/python310/bin/python3 -m venv /w/20251/oleksii/venv_depthsplat
source /w/20251/oleksii/venv_depthsplat/bin/activate
pip install --upgrade pip wheel
```

---

## 2. Install fixed diff-gaussian-rasterization

```bash
cd ~/code/
git clone https://github.com/dcharatan/diff-gaussian-rasterization-modified
cd diff-gaussian-rasterization-modified
```

Edit the file to fix compilation error:

```bash
nano cuda_rasterizer/rasterizer_impl.h
```

**At the very top** (near other `#include` lines), add:

```cpp
#include <cstdint>
```

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

Initialize submodules:

```bash
cd ~/code/diff-gaussian-rasterization-modified
git submodule update --init --recursive
```

Install the package:

```bash
pip install --no-build-isolation ../diff-gaussian-rasterization-modified
```

---

## 3. Clone and setup DepthSplat repository

```bash
cd ~/code
git clone https://github.com/cvg/depthsplat.git
cd depthsplat
```

Install PyTorch with CUDA 12.4 support:

```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
```

**Important**: Remove the original `diff-gaussian-rasterization` line from `requirements.txt` (we installed the fixed version manually):

```bash
sed -i '/diff-gaussian-rasterization/d' requirements.txt
```

Then install remaining dependencies:

```bash
pip install -r requirements.txt
```

---

## 4. Download pre-trained model and dataset (store on /w)

### Model
Download:
```
https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth
```

Final path:
```
/w/20251/oleksii/depthsplat/pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth
```

### Dataset (RE10K)
Download from Google Drive:
```
https://drive.google.com/file/d/1E1ui59ffUxXiDauDlrSCRKvB6qyQec7D/view?usp=drive_link
```

Unzip directly to:
```
/w/20251/oleksii/depthsplat/datasets/re10k/
```

Expected structure:
```
/w/20251/oleksii/depthsplat/datasets/re10k/test/
```

### Create symlinks in the repo

```bash
cd ~/code/depthsplat
ln -s /w/20251/oleksii/depthsplat/pretrained pretrained
ln -s /w/20251/oleksii/depthsplat/datasets datasets
```

---

## 5. Test run on RealEstate10K

Create the evaluation script:

```bash
cat > evaluate_real_estate.sh << 'EOF'
#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
    dataset.test_chunk_interval=1 \
    model.encoder.upsample_factor=4 \
    model.encoder.lowest_feature_resolution=4 \
    checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth \
    mode=test \
    dataset/view_sampler=evaluation \
    test.save_image=true \
    test.save_depth=true \
    output_dir=outputs/small_model_images_and_depth
EOF

chmod +x evaluate_real_estate.sh
```

Run it:

```bash
sh evaluate_real_estate.sh
```

**Outputs will appear in:**
```
~/code/depthsplat/outputs/small_model_images_and_depth
```

You should see rendered images and depth maps for RealEstate10K test scenes.

---

**Done!** You now have a working DepthSplat installation with the small pretrained model on HPC.