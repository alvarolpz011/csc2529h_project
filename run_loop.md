> **Notes:**
Everytime the bash script is executed, the dirs that say run_{run_number} will have to increase one. So the bash should check the last run number folder and increase it by one automatically.

1. Set run_number to be +1 from the max numebered run folder in outputs/depthsplat/dl3dv_run_1/
2. Set step_number to 1 for the first step of each run.
3. conda activate depthsplat_difix 
4. cd depthsplat
5. python src/scripts/convert_dl3dv_test.py     --input_dir /w/20251/your_user_dir/datasets     --output_dir datasets/dl3dv     --img_subdir images_8     --n_test 1
6. python src/scripts/generate_dl3dv_index.py --path /u/alvarolopez/Documents/csc2529/csc2529h_project/depthsplat/datasets/dl3dv/
7. CUDA_VISIBLE_DEVICES=0 python -m src.main \
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
    output_dir=../outputs/depthsplat/dl3dv_run_1/step1
8. Rename output images to "frame_{original_number+1}.png".
8. cd ..
9. python feed_depthsplat_out_to_diffix.py   --input_dir "/u/alvarolopez/Documents/csc2529/csc2529h_project/outputs/depthsplat/dl3dv_run_1/step1/images"   --output_dir "/u/alvarolopez/Documents/csc2529/csc2529h_project/outputs/depthsplat_difix/d3lv_run_1/step1"
10. Copy all contents inside the images folder into the previous step1 dir. 
11. delete the images folder inside step1 dir.
10. FOR each hash in "/u/alvarolopez/Documents/csc2529/csc2529h_project/outputs/depthsplat_difix/d3lv_run_1/step1", inside <hash> dir, mkdir /nerfstudio/images_8 
11. move all .png files into the created "<hash>/nerfstudio/images_8" direcrtory
12. Rename all images inside "<hash>/nerfstudio/images_8" to "frame_{original_number+1}.png" if needed
13. Copy transforms.json from original input_dir in step 5 (/w/20251/your_user_dir/datasets), inside datasets/<hash>/nerfstudio
/nerfstudio/images_8" for each hash folder there into the difix output directory + <hash> + /nerfstudio

> **Notes:**
- Change the `input_dir` parameter to change numbers automatically for different runs.
- Change the `output_dir` parameter to change numbers automatically for different runs.

### Loop Starts
2. cd depthsplat
3. python src/scripts/convert_dl3dv_diffix_test.py --input_dir /u/alvarolopez/Documents/csc2529/csc2529h_project/outputs/depthsplat_difix/d3lv_run_{run_number}/step{step_number} --output_dir datasets/dl3dv --img_subdir images_8 --n_test 1 
4. python src/scripts/generate_dl3dv_index.py --path /u/alvarolopez/Documents/csc2529/csc2529h_project/depthsplat/datasets/dl3dv/
1. Set step_number to +1.
5. CUDA_VISIBLE_DEVICES=0 python -m src.main \
    +experiment=dl3dv \
    dataset.test_chunk_interval=1 \
    dataset.roots=[datasets/dl3dv] \
    dataset.image_shape=[256,448] \
    dataset.ori_image_shape=[256,448] \
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
    output_dir=../outputs/depthsplat/dl3dv_run_{run_number}/step{step_number}

6. cd ..

9. python feed_depthsplat_out_to_diffix.py   --input_dir "/u/alvarolopez/Documents/csc2529/csc2529h_project/outputs/depthsplat/dl3dv_run_{run_number}/step{step_number}/images"   --output_dir "/u/alvarolopez/Documents/csc2529/csc2529h_project/outputs/depthsplat_difix/d3lv_run_{run_number}/step{step_number}"
10. Copy all contents inside the images folder into the previous step{step_number} dir. 
11. delete the images folder inside step{step_number} dir.
10. FOR each hash in "/u/alvarolopez/Documents/csc2529/csc2529h_project/outputs/depthsplat_difix/d3lv_run_{run_number}/step{step_number}", inside <hash> dir, mkdir /nerfstudio/images_8 
11. move all .png files into the created "<hash>/nerfstudio/images_8" direcrtory
12. Rename all images inside "<hash>/nerfstudio/images_8" to "frame_{original_number+1}.png" if needed
13. Copy transforms.json from original input_dir in step 5 (/w/20251/your_user_dir/datasets), inside datasets/<hash>/nerfstudio
/nerfstudio/images_8" for each hash folder there into the difix output directory + <hash> + /nerfstudio