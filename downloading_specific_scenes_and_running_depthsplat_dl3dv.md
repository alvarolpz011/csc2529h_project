# Instructions for Downloading and Preparing Subset of DL3DV Benchmark Dataset

This assumes you already have a working Python environment, and a `pretrained` and `datasets` directory (preferrably using Oleksii's DepthSplat setup instructions if using CS slurm machines).
These instructions guide you through downloading specific scenes from the DL3DV benchmark dataset and preparing them for use in DepthSplat.

---

## 1. Download Specific Scenes

> **Note:** It is suggested to use the `--clean_cache` flag, as it saves space by cleaning the cache folder created by the Huggingface Hub API.

The download.py script will be in the `scripts` folder. You have to add your Hugging Face token inside download.py to authenticate.

### Example Commands

```bash
# Download scene with hash 0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695
python download.py \
  --subset hash \
  --hash 0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695 \
  --odir /w/20251/alvarolopez/datasets

# Download scene with hash 9641a1ed7963ce5ca734cff3e6ccea3dfa8bcb0b0a3ff78f65d32a080de2d71e
python download.py \
  --subset hash \
  --hash 9641a1ed7963ce5ca734cff3e6ccea3dfa8bcb0b0a3ff78f65d32a080de2d71e \
  --odir /w/20251/alvarolopez/datasets
```

> **Notes:**
> * Each scene may take ~40 minutes to download at 10 MB/s.
> * You must add your Hugging Face token inside download.py to authenticate.
> * Full list of scene hashes can be found [here](https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark/blob/main/benchmark-meta.csv).

## 2. Convert Scenes to Test Dataset

Once downloaded, run the following script to generate the test data:

```bash
python src/scripts/convert_dl3dv_test.py \
    --input_dir /w/20251/alvarolopez/datasets \
    --output_dir datasets/dl3dv/test \
    --img_subdir images_8 \
    --n_test 1
```
## 3. Update Dataset Index

Edit generate_dl3dv_index.py to point to your dataset location:
```bash
DATASET_PATH = Path("/u/alvarolopez/Documents/csc2529/depthsplat/depthsplat/datasets/dl3dv/")
```

## 4. Download Pretrained Model

Use wget to download the DepthSplat pretrained model into the pretrained directory:

```bash
wget -P pretrained https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth
```

## 5. Run DepthSplat for Single Scene Testing

Run the model on the downloaded scene:
```bash
CUDA_VISIBLE_DEVICES=0 python -m src.main \
    +experiment=dl3dv \
    dataset.test_chunk_interval=1 \
    dataset.roots=[datasets/dl3dv] \
    dataset.image_shape=[256,448] \
    dataset.ori_image_shape=[270,480] \
    model.encoder.num_scales=2 \
    model.encoder.upsample_factor=4 \
    model.encoder.lowest_feature_resolution=8 \
    model.encoder.monodepth_vit_type=vitb \
    checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth \
    mode=test \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=6 \
    dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
    test.save_image=true \
    test.save_depth=true \
    test.save_depth_npy=true \
    test.save_gaussian=true \
    output_dir=outputs/dl3dv_single_scene
```

> **Notes:**
> * This uses one of the existing JSON indexes. It may be possible to create your own index if needed. For now it seems their premade indixes should work fine.
> * Adjust parameters like `dataset.view_sampler.index_path` and `output_dir` as required.
> The view sampler index file defines, for each possible scene of the complete DL3DV dataset, which frames (views) are to be used as context views and what the target views will be. The target views are the ground truth that Depthsplat will try to render as novel views as close as possible. The PSNR and other metrix are computed based on these target views and the rendered views. The context views are the ones that will have the depthmaps computed by depthsplat. 



This will output:
* Rendered images of the novel views under the directory `outputs/dl3dv_single_scene/images/<scene_hash>/color/`.
* Predicted depth maps under the directory `outputs/dl3dv_single_scene/images/<scene_hash>/depth/`.
* `PLY` files of the learned Gaussian splatting representation as file named: `outputs/dl3dv_single_scene/gaussians/<scene_hash>.ply`.
