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

# Download data
python3 src/download_data.py --dataset lego
```

-----

## Running All Experiments

To run the experiments, use the following command:

```bash
bash run_experiment.sh
```

## Running Individual Experiments

To run individual experiments, use the commands below.

### **Train Difix (Base, Clean)**

```bash
accelerate launch --mixed_precision=bf16 src/train_difix.py \
    --output_dir=./outputs/lego/clean/base \
    --dataset_path="data/lego" \
    --max_train_steps 2000 \
    --num_training_epochs 50 \
    --resolution=512 \
    --learning_rate 1e-4 \
    --train_batch_size=1 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 \
    --tracker_project_name "difix" \
    --tracker_run_name "lego-clean-base"
```

### **Train Difix (Depth-based, Clean)**

```bash
accelerate launch --mixed_precision=bf16 src/train_difix.py \
    --output_dir=./outputs/lego/clean/depth \
    --dataset_path="data/lego" \
    --max_train_steps 2000 \
    --num_training_epochs 50 \
    --resolution=512 \
    --learning_rate 1e-4 \
    --train_batch_size=1 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 \
    --tracker_project_name "difix" \
    --tracker_run_name "lego-clean-depth" \
    --use_depth
```

### **Train Difix (Canny-based, Clean)**

```bash
accelerate launch --mixed_precision=bf16 src/train_difix.py \
    --output_dir=./outputs/lego/clean/canny \
    --dataset_path="data/lego" \
    --max_train_steps 2000 \
    --num_training_epochs 50 \
    --resolution=512 \
    --learning_rate 1e-4 \
    --train_batch_size=1 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 \
    --tracker_project_name "difix" \
    --tracker_run_name "lego-clean-canny" \
    --use_canny
```

### **Train Difix (Depth and Canny, Clean)**

```bash
accelerate launch --mixed_precision=bf16 src/train_difix.py \
    --output_dir=./outputs/lego/clean/depth_canny \
    --dataset_path="data/lego" \
    --max_train_steps 2000 \
    --num_training_epochs 50 \
    --resolution=512 \
    --learning_rate 1e-4 \
    --train_batch_size=1 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 \
    --tracker_project_name "difix" \
    --tracker_run_name "lego-clean-depth_canny" \
    --use_depth \
    --use_canny
```

## **Evaluate Metrics (Clean)**

```bash
python3 src/eval_metrics.py \
    --dataset_path data/lego \
    --checkpoint_path outputs/lego/clean/base/checkpoints/model_2000.pkl \
    --run_name "lego-clean-base"
```