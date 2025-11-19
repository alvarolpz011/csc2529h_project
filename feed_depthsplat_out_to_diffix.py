import os
import argparse
from Difix3D.src.pipeline_difix import DifixPipeline
from diffusers.utils import load_image

# Example execution:
# python feed_depthsplat_out_to_diffix.py   --input_dir "/u/alvarolopez/Documents/csc2529/csc2529h_project/depthsplat/outputs/dl3dv_single_scene/images"   --output_dir "/u/alvarolopez/Documents/csc2529/csc2529h_project/outputs/depthsplat_difix"

def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Run Difix pipeline on DepthSplat output scenes.")
    
    # Input directory argument
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True, 
        help="Path to the DepthSplat output directory (e.g., /path/to/dl3dv_single_scene/images)"
    )
    
    # Output directory argument (optional, but recommended to avoid hardcoding)
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./outputs/depthsplat_difix",
        help="Path to save the processed images"
    )

    args = parser.parse_args()

    # 2. Assign variables from arguments
    input_root = args.input_dir
    output_root = args.output_dir

    # 3. Load Model
    print("Loading DifixPipeline...")
    pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    pipe.to("cuda")

    # 4. Process Scenes
    scenes_hashes = os.listdir(input_root)

    for scene in scenes_hashes:
        # Construct path to the 'color' folder
        color_dir = os.path.join(input_root, scene, "color")
        
        # Skip if not a directory or if 'color' folder doesn't exist
        if not os.path.isdir(os.path.join(input_root, scene)) or not os.path.exists(color_dir):
            continue

        print(f"Processing scene: {scene}")
        
        for image_name in sorted(os.listdir(color_dir)):
            image_path = os.path.join(color_dir, image_name)
            
            # Load image
            try:
                input_image = load_image(image_path)
            except Exception as e:
                print(f"Skipping {image_name}: {e}")
                continue

            prompt = "remove degradation"
            
            # Run inference
            output_image = pipe(
                prompt, 
                image=input_image, 
                num_inference_steps=1, 
                timesteps=[199], 
                guidance_scale=0.0
            ).images[0]

            # Prepare output directory
            scene_output_dir = os.path.join(output_root, scene)
            os.makedirs(scene_output_dir, exist_ok=True)

            # Save image
            save_path = os.path.join(scene_output_dir, image_name)
            output_image.save(save_path)
            print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()