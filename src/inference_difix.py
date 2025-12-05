import os
import imageio
import argparse
import numpy as np
import cv2
from PIL import Image
from glob import glob
from tqdm import tqdm
import torch
from torchvision import transforms
from model import Difix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input image or directory')
    parser.add_argument('--ref_image', type=str, default=None, help='Path to the reference image or directory')
    parser.add_argument('--depth_image', type=str, default=None, help='Path to the depth image or directory')
    parser.add_argument('--canny_image', type=str, default=None, help='Path to the canny image or directory. If None and use_canny is set, computed from input.')
    parser.add_argument('--height', type=int, default=512, help='Height of the input image')
    parser.add_argument('--width', type=int, default=512, help='Width of the input image')
    parser.add_argument('--prompt', type=str, required=True, help='The prompt to be used')
    parser.add_argument('--model_name', type=str, default=None, help='Name of the pretrained model')
    parser.add_argument('--model_path', type=str, default=None, help='Path to a model state dict')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timestep', type=int, default=199)
    parser.add_argument('--video', action='store_true')
    
    # Adapter Flags
    parser.add_argument('--use_depth', action='store_true', help='Initialize model with depth adapter')
    parser.add_argument('--use_canny', action='store_true', help='Initialize model with canny adapter')
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model with flags
    model = Difix(
        pretrained_name=args.model_name,
        pretrained_path=args.model_path,
        timestep=args.timestep,
        mv_unet=True if args.ref_image is not None else False,
        use_depth_adapter=args.use_depth,
        use_canny_adapter=args.use_canny
    )
    model.set_eval()

    # Load images
    if os.path.isdir(args.input_image):
        input_images = sorted(glob(os.path.join(args.input_image, "*.png")))
    else:
        input_images = [args.input_image]

    # Load refs
    if args.ref_image:
        if os.path.isdir(args.ref_image):
            ref_images = sorted(glob(os.path.join(args.ref_image, "*")))
        else:
            ref_images = [args.ref_image]
        assert len(input_images) == len(ref_images)

    # Load depths
    depth_images = None
    if args.depth_image:
        if os.path.isdir(args.depth_image):
            depth_images = sorted(glob(os.path.join(args.depth_image, "*")))
        else:
            depth_images = [args.depth_image]
        assert len(input_images) == len(depth_images)

    # Load Canny (Explicit)
    canny_images = None
    if args.canny_image:
        if os.path.isdir(args.canny_image):
            canny_images = sorted(glob(os.path.join(args.canny_image, "*")))
        else:
            canny_images = [args.canny_image]
        assert len(input_images) == len(canny_images)

    # Process
    output_images = []
    for i, input_path in enumerate(tqdm(input_images, desc="Processing")):
        image = Image.open(input_path).convert('RGB')
        ref = Image.open(ref_images[i]).convert('RGB') if args.ref_image else None
        
        # Load depth map
        depth = None
        if depth_images:
            depth = Image.open(depth_images[i]).convert('L')

        # Handle Canny
        canny = None
        if args.use_canny:
            if canny_images:
                canny = Image.open(canny_images[i]).convert('L')
            else:
                # Compute on the fly
                image_np = np.array(image)
                edges = cv2.Canny(image_np, 100, 200)
                canny = Image.fromarray(edges).convert("L")

        output_image = model.sample(
            image,
            height=args.height,
            width=args.width,
            ref_image=ref,
            depth_map=depth, 
            canny_map=canny,
            prompt=args.prompt
        )
        output_images.append(output_image)

    # Save outputs
    if args.video:
        video_path = os.path.join(args.output_dir, "output.mp4")
        writer = imageio.get_writer(video_path, fps=30)
        for out_img in output_images:
            writer.append_data(np.array(out_img))
        writer.close()
    else:
        for i, out_img in enumerate(output_images):
            out_img.save(os.path.join(args.output_dir, os.path.basename(input_images[i])))