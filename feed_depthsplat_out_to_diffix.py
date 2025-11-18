import os
from Difix3D.src.pipeline_difix import DifixPipeline
from diffusers.utils import load_image

DEPTHSPLAT_DL3DV_OUTPUT_DIR = '/u/alvarolopez/Documents/csc2529/depthsplat/depthsplat/outputs/dl3dv_single_scene/images'
SCENES_HASHES = os.listdir(DEPTHSPLAT_DL3DV_OUTPUT_DIR)

pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe.to("cuda")

for scene in SCENES_HASHES:
    for image in sorted(os.listdir(f"{DEPTHSPLAT_DL3DV_OUTPUT_DIR}/{scene}/color")):
        input_image = load_image(f"{DEPTHSPLAT_DL3DV_OUTPUT_DIR}/{scene}/color/{image}")
        prompt = "remove degradation"
        output_image = pipe(prompt, image=input_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]
        if not os.path.exists(f"/u/alvarolopez/Documents/csc2529/csc2529h_project/outputs/depthsplat_difix/{scene}"):
            os.makedirs(f"/u/alvarolopez/Documents/csc2529/csc2529h_project/outputs/depthsplat_difix/{scene}")
        output_image.save(f"/u/alvarolopez/Documents/csc2529/csc2529h_project/outputs/depthsplat_difix/{scene}/{image}")


