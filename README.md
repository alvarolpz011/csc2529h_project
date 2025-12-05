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
mkdir -p data outputs

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

### **Train Difix (Base)**

```bash
accelerate launch --mixed_precision=bf16 src/train_difix.py \
    --output_dir=./outputs/lego/degraded/base \
    --dataset_path="data/lego" \
    --max_train_steps 2000 \
    --num_training_epochs 50 \
    --resolution=512 \
    --learning_rate 1e-4 \
    --train_batch_size=1 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 \
    --tracker_project_name "difix" \
    --tracker_run_name "lego-degraded-base" \
    --degrade_inputs
```

### **Train Difix (Depth-based)**

```bash
accelerate launch --mixed_precision=bf16 src/train_difix.py \
    --output_dir=./outputs/lego/degraded/depth \
    --dataset_path="data/lego" \
    --max_train_steps 2000 \
    --num_training_epochs 50 \
    --resolution=512 \
    --learning_rate 1e-4 \
    --train_batch_size=1 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 \
    --tracker_project_name "difix" \
    --tracker_run_name "lego-degraded-depth" \
    --use_depth \
    --degrade_inputs
```

### **Train Difix (Canny-based)**

```bash
accelerate launch --mixed_precision=bf16 src/train_difix.py \
    --output_dir=./outputs/lego/degraded/canny \
    --dataset_path="data/lego" \
    --max_train_steps 2000 \
    --num_training_epochs 50 \
    --resolution=512 \
    --learning_rate 1e-4 \
    --train_batch_size=1 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 \
    --tracker_project_name "difix" \
    --tracker_run_name "lego-degraded-canny" \
    --use_canny \
    --degrade_inputs
```

### **Train Difix (Depth and Canny)**

```bash
accelerate launch --mixed_precision=bf16 src/train_difix.py \
    --output_dir=./outputs/lego/degraded/depth_canny \
    --dataset_path="data/lego" \
    --max_train_steps 2000 \
    --num_training_epochs 50 \
    --resolution=512 \
    --learning_rate 1e-4 \
    --train_batch_size=1 \
    --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 \
    --tracker_project_name "difix" \
    --tracker_run_name "lego-degraded-depth_canny" \
    --use_depth \
    --use_canny \
    --degrade_inputs
```

## **Evaluate Metrics**

```bash
python3 src/eval_metrics.py \
    --dataset_path data/lego \
    --checkpoint_path outputs/lego/degraded/base/checkpoints/model_2000.pkl \
    --run_name "lego-degraded-base" \
    --degrade_inputs
```