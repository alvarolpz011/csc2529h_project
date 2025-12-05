# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional, Type
from pathlib import Path
from PIL import Image
import os
import tqdm
import random
import numpy as np
import cv2
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs

from difix3d.difix3d_datamanager import Difix3DDataManagerConfig
import sys
sys.path.append(os.getcwd())
from src.model import Difix
from examples.utils import CameraPoseInterpolator


@dataclass
class Difix3DPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: Difix3DPipeline)
    datamanager: Difix3DDataManagerConfig = Difix3DDataManagerConfig()
    
    steps_per_fix: int = 2000
    steps_per_val: int = 5000
    
    use_depth_adapter: bool = True
    use_canny_adapter: bool = True

    diffusion_ckpt_path: Path = Path("outputs/materials/degraded/base/checkpoints/model_2000.pkl")


class Difix3DPipeline(VanillaPipeline):
    """Difix3D pipeline"""

    config: Difix3DPipelineConfig

    def __init__(
        self,
        config: Difix3DPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
        render_dir: str = "renders",
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)

        self.render_dir = render_dir
        
        print(f"Initializing Difix | Depth: {self.config.use_depth_adapter} | Canny: {self.config.use_canny_adapter}")
        
        self.difix = Difix(
            timestep=199, 
            use_depth_adapter=self.config.use_depth_adapter,
            use_canny_adapter=self.config.use_canny_adapter,
            mv_unet=False
        )
        self.difix.to("cpu")

        ckpt_path = self.config.diffusion_ckpt_path

        if os.path.exists(ckpt_path):
            print(f"Loading Refinement Model from {ckpt_path}")
            sd = torch.load(ckpt_path, map_location="cpu")
            
            if "state_dict_vae" in sd:
                _sd_vae = self.difix.vae.state_dict()
                for k in sd["state_dict_vae"]:
                    if k in _sd_vae: _sd_vae[k] = sd["state_dict_vae"][k]
                self.difix.vae.load_state_dict(_sd_vae)

            if "state_dict_unet" in sd:
                self.difix.unet.load_state_dict(sd["state_dict_unet"])

            if self.config.use_depth_adapter and "state_dict_adapter" in sd:
                print("Loading Depth Adapter...")
                self.difix.adapter.load_state_dict(sd["state_dict_adapter"])
            
            if self.config.use_canny_adapter and "state_dict_adapter_canny" in sd:
                print("Loading Canny Adapter...")
                self.difix.adapter_canny.load_state_dict(sd["state_dict_adapter_canny"])
            
            self.difix.set_eval()
        else:
            print(f"WARNING: Checkpoint {ckpt_path} not found!")

        self.training_poses = self.datamanager.train_dataparser_outputs.cameras.camera_to_worlds
        self.training_poses = torch.cat([
            self.training_poses, 
            torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(self.training_poses.shape[0], 1, 1)
        ], dim=1)
        
        self.testing_poses = self.datamanager.dataparser.get_dataparser_outputs(split=self.datamanager.test_split).cameras.camera_to_worlds
        self.testing_poses = torch.cat([
            self.testing_poses, 
            torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).repeat(self.testing_poses.shape[0], 1, 1)
        ], dim=1)
        
        self.current_novel_poses = self.training_poses
        self.current_novel_cameras = self.datamanager.train_dataparser_outputs.cameras
        self.interpolator = CameraPoseInterpolator(rotation_weight=1.0, translation_weight=1.0)
        self.novel_datamanagers = []

    def get_train_loss_dict(self, step: int):
        if len(self.novel_datamanagers) == 0 or random.random() < 0.6:
            ray_bundle, batch = self.datamanager.next_train(step)
        else:
            ray_bundle, batch = self.novel_datamanagers[-1].next_train(step)

        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        if step >= 1000 and (step % self.config.steps_per_fix == 0):
            self.fix(step)

        if (step % self.config.steps_per_val == 0):
            self.val(step)

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def render_traj(self, step, cameras, tag="novel"):
        rgb_dir = f"{self.render_dir}/{tag}/{step}/Pred"
        depth_dir = f"{self.render_dir}/{tag}/{step}/Depth"
        os.makedirs(rgb_dir, exist_ok=True)
        
        if self.config.use_depth_adapter:
            os.makedirs(depth_dir, exist_ok=True)

        for i in tqdm.trange(0, len(cameras), desc="Rendering trajectory"):
            with torch.no_grad():
                outputs = self.model.get_outputs_for_camera(cameras[i])

            rgb = outputs['rgb']
            rgb_path = f"{rgb_dir}/{i:04d}.png"
            rgb_canvas = (rgb.cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(rgb_canvas).save(rgb_path)

            if self.config.use_depth_adapter:
                if 'depth' in outputs:
                    depth = outputs['depth']
                elif 'expected_depth' in outputs:
                    depth = outputs['expected_depth']
                else:
                    continue

                depth_min, depth_max = depth.min(), depth.max()
                if depth_max - depth_min > 1e-6:
                    depth_norm = (depth - depth_min) / (depth_max - depth_min)
                else:
                    depth_norm = depth

                depth_path = f"{depth_dir}/{i:04d}.png"
                depth_canvas = (depth_norm.squeeze().cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(depth_canvas, mode='L').save(depth_path)

    @torch.no_grad()
    def val(self, step):
        cameras = self.datamanager.dataparser.get_dataparser_outputs(split=self.datamanager.test_split).cameras
        for i in tqdm.trange(0, len(cameras), desc="Running evaluation"):
            with torch.no_grad():
                outputs = self.model.get_outputs_for_camera(cameras[i])
            rgb_path = f"{self.render_dir}/val/{step}/{i:04d}.png"
            os.makedirs(os.path.dirname(rgb_path), exist_ok=True)
            rgb_canvas = outputs['rgb'].cpu().numpy()
            rgb_canvas = (rgb_canvas * 255).astype(np.uint8)
            Image.fromarray(rgb_canvas).save(rgb_path)

    @torch.no_grad()
    def fix(self, step: int):
        print(f"[Step {step}] Running Fixer...")
        self.difix.to("cpu")
        torch.cuda.empty_cache()

        novel_poses = self.interpolator.shift_poses(self.current_novel_poses.numpy(), self.testing_poses.numpy(), distance=0.5)
        novel_poses = torch.from_numpy(novel_poses).to(self.testing_poses.dtype)

        cameras = self.datamanager.train_dataparser_outputs.cameras
        if cameras.distortion_params is not None:
            dist_params = cameras.distortion_params[0].repeat(len(novel_poses), 1)
        else:
            dist_params = None

        cameras = Cameras(
            fx=cameras.fx[0].repeat(len(novel_poses), 1),
            fy=cameras.fy[0].repeat(len(novel_poses), 1),
            cx=cameras.cx[0].repeat(len(novel_poses), 1),
            cy=cameras.cy[0].repeat(len(novel_poses), 1),
            distortion_params=dist_params,
            height=cameras.height[0].repeat(len(novel_poses), 1),
            width=cameras.width[0].repeat(len(novel_poses), 1),
            camera_to_worlds=novel_poses[:, :3, :4],
            camera_type=cameras.camera_type[0].repeat(len(novel_poses), 1),
            metadata=cameras.metadata,
        )

        self.render_traj(step, cameras)

        ref_image_indices = self.interpolator.find_nearest_assignments(self.training_poses.numpy(), novel_poses.numpy())
        ref_image_filenames = np.array(self.datamanager.train_dataparser_outputs.image_filenames)[ref_image_indices].tolist()

        image_filenames = []
        pred_dir = f"{self.render_dir}/novel/{step}/Pred"
        fixed_dir = f"{self.render_dir}/novel/{step}/Fixed"
        ref_dir = f"{self.render_dir}/novel/{step}/Ref"
        os.makedirs(fixed_dir, exist_ok=True)
        os.makedirs(ref_dir, exist_ok=True)

        self.difix.to(self.device)

        for i in tqdm.trange(0, len(novel_poses), desc="Refining..."):
            rgb_path = f"{pred_dir}/{i:04d}.png"
            image_pil = Image.open(rgb_path).convert("RGB")
            ref_pil = Image.open(ref_image_filenames[i]).convert("RGB")

            depth_pil = None
            if self.config.use_depth_adapter:
                depth_path = f"{self.render_dir}/novel/{step}/Depth/{i:04d}.png"
                depth_pil = Image.open(depth_path).convert("L")

            canny_pil = None
            if self.config.use_canny_adapter:
                image_np = np.array(image_pil)
                edges = cv2.Canny(image_np, 100, 200)
                canny_pil = Image.fromarray(edges).convert("L")

            output_image = self.difix.sample(
                image=image_pil,
                width=image_pil.width,
                height=image_pil.height,
                ref_image=ref_pil,
                depth_map=depth_pil,
                canny_map=canny_pil,
                prompt="high quality 3d render"
            )

            output_path = f"{fixed_dir}/{i:04d}.png"
            output_image.save(output_path)
            image_filenames.append(Path(output_path))
            
            if ref_pil is not None:
                ref_pil.save(f"{ref_dir}/{i:04d}.png")

        self.difix.to("cpu")
        torch.cuda.empty_cache()

        dataparser_outputs = self.datamanager.train_dataparser_outputs
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=dataparser_outputs.scene_box,
            mask_filenames=None,
            dataparser_scale=dataparser_outputs.dataparser_scale,
            dataparser_transform=dataparser_outputs.dataparser_transform,
            metadata=dataparser_outputs.metadata,
        )

        datamanager_config = Difix3DDataManagerConfig(
            dataparser=self.config.datamanager.dataparser,
            train_num_rays_per_batch=self.config.datamanager.train_num_rays_per_batch,
            eval_num_rays_per_batch=self.config.datamanager.eval_num_rays_per_batch,
        )
        
        datamanager = datamanager_config.setup(
            device=self.datamanager.device, 
            test_mode=self.datamanager.test_mode, 
            world_size=self.datamanager.world_size, 
            local_rank=self.datamanager.local_rank
        )

        datamanager.train_dataparser_outputs = dataparser_outputs
        datamanager.train_dataset = datamanager.create_train_dataset()
        datamanager.setup_train()

        self.novel_datamanagers.append(datamanager)
        self.current_novel_poses = novel_poses
        self.current_novel_cameras = cameras