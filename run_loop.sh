#!/bin/bash

# Configuration
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading configuration from $ENV_FILE"
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.* ]] || [[ -z "$key" ]] && continue
        export "$key"="$(echo "$value" | tr -d '"')"
    done < "$ENV_FILE"
else
    echo "Error: $ENV_FILE not found. Exiting."
    exit 1
fi

# Step 1: Determine run_number
echo "Finding the latest run number..."
RUN_DIR="${DEPTHSPLAT_OUTPUTS_BASE}/dl3dv_run_1"
mkdir -p "$RUN_DIR"

# Find max run number
max_run=0
for dir in "$RUN_DIR"/run_*; do
  if [ -d "$dir" ]; then
    run_num=$(basename "$dir" | sed 's/run_//')
    if [[ "$run_num" =~ ^[0-9]+$ ]] && [ "$run_num" -gt "$max_run" ]; then
      max_run=$run_num
    fi
  fi
done

RUN_NUMBER=$((max_run + 1))
STEP_NUMBER=1

echo "Run number: $RUN_NUMBER"
echo "Step number: $STEP_NUMBER"

# Create run directory
RUN_PATH="${RUN_DIR}/run_${RUN_NUMBER}"
mkdir -p "$RUN_PATH"

# Step 2: Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

# Step 3: Navigate to depthsplat directory
cd "$DEPTHSPLAT_DIR" || exit 1

# Step 4: Convert dataset
echo "Converting dataset..."
python src/scripts/convert_dl3dv_test.py \
  --input_dir "$ORIGINAL_DOWNLOADED_DATASET_DIR" \
  --output_dir datasets/dl3dv \
  --img_subdir images_8 \
  --n_test 1

# Step 5: Generate index
echo "Generating dl3dv index..."
python src/scripts/generate_dl3dv_index.py \
  --path "$DEPTHSPLAT_DIR/datasets/dl3dv/"

# Step 6: Run depthsplat model
echo "Running depthsplat model..."
OUTPUT_STEP_DIR="${RUN_PATH}/step${STEP_NUMBER}"
mkdir -p "$OUTPUT_STEP_DIR"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -m src.main \
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
  output_dir="$OUTPUT_STEP_DIR"

# Step 7: Rename output images
echo "Renaming output images..."
if [ -d "${OUTPUT_STEP_DIR}/images" ]; then
  for img in "${OUTPUT_STEP_DIR}/images"/*.png; do
    if [ -f "$img" ]; then
      basename_img=$(basename "$img")
      original_num="${basename_img%.png}"
      original_num="${original_num##*_}"
      if [[ "$original_num" =~ ^[0-9]+$ ]]; then
        new_num=$((original_num + 1))
        mv "$img" "${OUTPUT_STEP_DIR}/images/frame_${new_num}.png"
      fi
    fi
  done
fi

echo "Calculating TRUE metrics for Step ${STEP_NUMBER}..."
cd ..
python scripts/evaluate_step.py \
    --pred_dir "${OUTPUT_STEP_DIR}/images" \
    --gt_root "$ORIGINAL_DOWNLOADED_DATASET_DIR"
cd "$DEPTHSPLAT_DIR" || exit 1


# Step 8: Navigate to parent directory
cd .. || exit 1

# Step 9: Run difix script
echo "Running feed_depthsplat_out_to_diffix.py..."
DIFIX_RUN_DIR="${DIFIX_OUTPUTS_BASE}/d3lv_run_${RUN_NUMBER}"
mkdir -p "$DIFIX_RUN_DIR/step${STEP_NUMBER}"

python feed_depthsplat_out_to_diffix.py \
  --input_dir "${OUTPUT_STEP_DIR}/images" \
  --output_dir "${DIFIX_RUN_DIR}/step${STEP_NUMBER}"

# Step 10: Copy images to step directory and clean up
echo "Copying and cleaning up image directories..."
if [ -d "${DIFIX_RUN_DIR}/step${STEP_NUMBER}/images" ]; then
  cp -r "${DIFIX_RUN_DIR}/step${STEP_NUMBER}/images"/* "${DIFIX_RUN_DIR}/step${STEP_NUMBER}/"
  rm -rf "${DIFIX_RUN_DIR}/step${STEP_NUMBER}/images"
fi

# Step 11: Process each hash directory
echo "Processing hash directories..."
for hash_dir in "${DIFIX_RUN_DIR}/step${STEP_NUMBER}"/*; do
  if [ -d "$hash_dir" ] && [ "$(basename "$hash_dir")" != "images" ]; then
    hash_name=$(basename "$hash_dir")
    
    # Create nerfstudio structure
    mkdir -p "${hash_dir}/nerfstudio/images_8"
    
    # Move PNG files
    if [ -d "$hash_dir" ]; then
      find "$hash_dir" -maxdepth 1 -name "*.png" -exec mv {} "${hash_dir}/nerfstudio/images_8/" \;
    fi
    

    echo "Renaming images to correct format..."
    # Rename images if needed
    for img in "${hash_dir}/nerfstudio/images_8"/*.png; do
      if [ -f "$img" ]; then
        basename_img=$(basename "$img")
        
        # Skip files that are already renamed to avoid double-processing
        case "$basename_img" in
          frame_[0-9]*.png) continue ;;
        esac

        original_num="${basename_img%.png}"
        original_num="${original_num##*_}"
        
        case "$original_num" in
          *[!0-9]* | "") continue ;;
          *)
            # 1. Calculate using Base 10 to avoid Octal error
            val=$((10#$original_num + 1))
            
            # 2. Format with 5 digits and leading zeros using printf
            # %05d means: Pad with 0, width of 5, decimal integer
            new_num=$(printf "%05d" "$val")
            
            mv "$img" "${hash_dir}/nerfstudio/images_8/frame_${new_num}.png"
            ;;
        esac
      fi
    done
    
    # Copy transforms.json
    echo "Copying transforms.json for hash: $hash_name..."
    if [ -f "${ORIGINAL_DOWNLOADED_DATASET_DIR}/${hash_name}/nerfstudio/transforms.json" ]; then
      cp "${ORIGINAL_DOWNLOADED_DATASET_DIR}/${hash_name}/nerfstudio/transforms.json" \
         "${hash_dir}/nerfstudio/"
    else
      echo "Warning: transforms.json not found for $hash_name"
    fi
  fi
done

# Resize images using linear interpolation
echo "Resizing images using modify_transforms.py..."
python scripts/linear_interpolation.py --input_dir "${DIFIX_RUN_DIR}/step${STEP_NUMBER}"




# LOOP STARTS
# Python equivalent: for i in range(5):
for ((i=0; i<5; i++)); do

    echo "--------------------- Iteration $((i+1)) ---------------------"

    # Run Python to change transform.json
    echo "Modifying transforms.json..."
    python scripts/modify_transforms.py --data-dir "${DIFIX_RUN_DIR}/step${STEP_NUMBER}"

    # Navigate to Depthsplat Directory
    cd "$DEPTHSPLAT_DIR" || exit 1

    # Convert dataset
    echo "Converting dataset..."
    python src/scripts/convert_dl3dv_test.py \
    --input_dir "$DIFIX_OUTPUTS_BASE/d3lv_run_${RUN_NUMBER}/step${STEP_NUMBER}" \
    --output_dir datasets/dl3dv \
    --img_subdir images_8 \
    --n_test 1

    # Generate index
    echo "Generating dl3dv index..."
    python src/scripts/generate_dl3dv_index.py \
    --path "$DEPTHSPLAT_DIR/datasets/dl3dv/"

    STEP_NUMBER=$((STEP_NUMBER + 1))

    # Step 6: Run depthsplat model
    echo "Running depthsplat model..."
    OUTPUT_STEP_DIR="${RUN_PATH}/step${STEP_NUMBER}"
    mkdir -p "$OUTPUT_STEP_DIR"

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
      output_dir="$OUTPUT_STEP_DIR"

    # Step 7: Rename output images
    echo "Renaming output images..."
    if [ -d "${OUTPUT_STEP_DIR}/images" ]; then
      for img in "${OUTPUT_STEP_DIR}/images"/*.png; do
        if [ -f "$img" ]; then
          basename_img=$(basename "$img")
          original_num="${basename_img%.png}"
          original_num="${original_num##*_}"
          if [[ "$original_num" =~ ^[0-9]+$ ]]; then
            new_num=$((original_num + 1))
            mv "$img" "${OUTPUT_STEP_DIR}/images/frame_${new_num}.png"
          fi
        fi
      done
    fi

    echo "Calculating TRUE metrics for Step ${STEP_NUMBER}..."
    cd ..
    python scripts/evaluate_step.py \
        --pred_dir "${OUTPUT_STEP_DIR}/images" \
        --gt_root "$ORIGINAL_DOWNLOADED_DATASET_DIR"
    
    if (( i == 4 )); then
        echo "Final iteration completed. Exiting loop."
        echo "Pipeline completed successfully!"
        echo "Output directory: $DIFIX_RUN_DIR"
        break
    fi

    # Step 9: Run difix script
    echo "Running feed_depthsplat_out_to_diffix.py..."
    DIFIX_RUN_DIR="${DIFIX_OUTPUTS_BASE}/d3lv_run_${RUN_NUMBER}"
    mkdir -p "$DIFIX_RUN_DIR/step${STEP_NUMBER}"

    python feed_depthsplat_out_to_diffix.py \
      --input_dir "${OUTPUT_STEP_DIR}/images" \
      --output_dir "${DIFIX_RUN_DIR}/step${STEP_NUMBER}"

    # Step 10: Copy images to step directory and clean up
    echo "Copying and cleaning up image directories..."
    if [ -d "${DIFIX_RUN_DIR}/step${STEP_NUMBER}/images" ]; then
      cp -r "${DIFIX_RUN_DIR}/step${STEP_NUMBER}/images"/* "${DIFIX_RUN_DIR}/step${STEP_NUMBER}/"
      rm -rf "${DIFIX_RUN_DIR}/step${STEP_NUMBER}/images"
    fi

    # Step 11: Process each hash directory
    echo "Processing hash directories..."
    for hash_dir in "${DIFIX_RUN_DIR}/step${STEP_NUMBER}"/*; do
      if [ -d "$hash_dir" ] && [ "$(basename "$hash_dir")" != "images" ]; then
        hash_name=$(basename "$hash_dir")
        
        # Create nerfstudio structure
        mkdir -p "${hash_dir}/nerfstudio/images_8"
        
        # Move PNG files
        if [ -d "$hash_dir" ]; then
          find "$hash_dir" -maxdepth 1 -name "*.png" -exec mv {} "${hash_dir}/nerfstudio/images_8/" \;
        fi
        

        echo "Renaming images to correct format..."
        # Rename images if needed
        for img in "${hash_dir}/nerfstudio/images_8"/*.png; do
          if [ -f "$img" ]; then
            basename_img=$(basename "$img")
            
            # Skip files that are already renamed to avoid double-processing
            case "$basename_img" in
              frame_[0-9]*.png) continue ;;
            esac

            original_num="${basename_img%.png}"
            original_num="${original_num##*_}"
            
            case "$original_num" in
              *[!0-9]* | "") continue ;;
              *)
                # 1. Calculate using Base 10 to avoid Octal error
                val=$((10#$original_num + 1))
                
                # 2. Format with 5 digits and leading zeros using printf
                # %05d means: Pad with 0, width of 5, decimal integer
                new_num=$(printf "%05d" "$val")
                
                mv "$img" "${hash_dir}/nerfstudio/images_8/frame_${new_num}.png"
                ;;
            esac
          fi
        done
        
        # Copy transforms.json
        echo "Copying transforms.json for hash: $hash_name..."
        if [ -f "${ORIGINAL_DOWNLOADED_DATASET_DIR}/${hash_name}/nerfstudio/transforms.json" ]; then
          cp "${ORIGINAL_DOWNLOADED_DATASET_DIR}/${hash_name}/nerfstudio/transforms.json" \
            "${hash_dir}/nerfstudio/"
        else
          echo "Warning: transforms.json not found for $hash_name"
        fi
      fi
    done

    # Resize images using linear interpolation
    echo "Resizing images using modify_transforms.py..."
    python scripts/linear_interpolation.py --input_dir "${DIFIX_RUN_DIR}/step${STEP_NUMBER}"

done