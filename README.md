# Difix3D

## 🛠️ Setup and Data Preparation

These commands set up the Conda environment, install dependencies, and prepare the necessary data and directory structure.

```bash
conda create -n difix python=3.10.14
conda activate difix
pip install -r requirements.txt
cd examples/nerfstudio
pip install -e .
cd ../../
mkdir -p /w/20252/<your_username>/Difix3D/data
ln -s /w/20252/<your_username>/Difix3D/data data
mkdir -p /w/20252/<your_username>/Difix3D/outputs
ln -s /w/20252/<your_username>/Difix3D/outputs outputs
python3 src/download_data.py
python3 src/create_json.py
```

-----

## 🚀 Training Without Degradation

This section details the training and evaluation steps using **clean** input data (without degradation).

### **Train Difix (Depth-based, Clean)**

```bash
accelerate launch --mixed_precision=bf16 src/train_difix.py \
    --output_dir=./outputs/difix_depth_clean/lego \
    --dataset_path="data/lego.json" \
    --max_train_steps 2000 \
    --num_training_epochs 50 \
    --resolution=512 \
    --learning_rate 1e-4 \
    --train_batch_size=1 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 \
    --tracker_project_name "difix-depth-clean" \
    --tracker_run_name "lego-depth-clean" \
    --use_depth
```

### **Train Difix (Standard, Clean)**

```bash
accelerate launch --mixed_precision=bf16 src/train_difix.py \
    --output_dir=./outputs/difix_standard_clean/lego \
    --dataset_path="data/lego.json" \
    --max_train_steps 2000 \
    --num_training_epochs 50 \
    --resolution=512 \
    --learning_rate 1e-4 \
    --train_batch_size=1 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 \
    --tracker_project_name "difix-depth-clean" \
    --tracker_run_name "lego-standard-clean"
```

### **Evaluate Metrics (Clean)**

```bash
python3 src/eval_metrics.py \
    --ckpt_standard outputs/difix_standard_clean/lego/checkpoints/model_2000.pkl \
    --ckpt_depth outputs/difix_depth_clean/lego/checkpoints/model_2000.pkl \
    --dataset_path data/lego.json
```

-----

## 🧪 Training With Degradation

This section details the training and evaluation steps using **degraded** input data.

### **Train Difix (Depth-based, Degraded)**

```bash
accelerate launch --mixed_precision=bf16 src/train_difix.py \
    --output_dir=./outputs/difix_depth_degraded/lego \
    --dataset_path="data/lego.json" \
    --max_train_steps 2000 \
    --num_training_epochs 50 \
    --resolution=512 \
    --learning_rate 1e-4 \
    --train_batch_size=1 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 \
    --tracker_project_name "difix-depth-degraded" \
    --tracker_run_name "lego-depth-degraded" \
    --use_depth \
    --degrade_inputs
```

### **Train Difix (Standard, Degraded)**

```bash
accelerate launch --mixed_precision=bf16 src/train_difix.py \
    --output_dir=./outputs/difix_standard_degraded/lego \
    --dataset_path="data/lego.json" \
    --max_train_steps 2000 \
    --num_training_epochs 50 \
    --resolution=512 \
    --learning_rate 1e-4 \
    --train_batch_size=1 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 \
    --tracker_project_name "difix-depth-degraded" \
    --tracker_run_name "lego-standard-degraded" \
    --degrade_inputs
```

### **Evaluate Metrics (Degraded)**

**Note:** The `--degrade_inputs` flag is crucial here to apply the same corruption during testing.

```bash
python3 src/eval_metrics.py \
    --ckpt_standard outputs/difix_standard_degraded/lego/checkpoints/model_2000.pkl \
    --ckpt_depth outputs/difix_depth_degraded/lego/checkpoints/model_2000.pkl \
    --dataset_path data/lego.json \
    --degrade_inputs
```