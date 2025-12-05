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

from pathlib import Path
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig, LoggingConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification

from difix3d.difix3d_datamanager import Difix3DDataManagerConfig
from difix3d.difix3d import Difix3DModelConfig
from difix3d.difix3d_pipeline import Difix3DPipelineConfig
from difix3d.difix3d_trainer import Difix3DTrainerConfig

common_optimizers = {
    "proposal_networks": {
        "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
    },
    "fields": {
        "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
    },
    "camera_opt": {
        "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
    },
}

difix3d_base_method = MethodSpecification(
    config=Difix3DTrainerConfig(
        method_name="difix3d-base",
        experiment_name="difix3d-base",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=Difix3DPipelineConfig(
            datamanager=Difix3DDataManagerConfig(
                dataparser=BlenderDataParserConfig(data=Path("data/materials")),
                train_num_rays_per_batch=16384,
                eval_num_rays_per_batch=4096,
            ),
            model=Difix3DModelConfig(
                eval_num_rays_per_chunk=32768,
                average_init_density=0.01,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            diffusion_ckpt_path=Path("outputs/materials/degraded/base/checkpoints/model_2000.pkl"),
            use_depth_adapter=False,
            use_canny_adapter=False,
            steps_per_fix=2000,
        ),
        optimizers=common_optimizers,
        viewer=ViewerConfig(num_rays_per_chunk=32768),
        vis="viewer+wandb",
        logging=LoggingConfig(steps_per_log=10),
    ),
    description="Difix3D Base (No Adapters)",
)

difix3d_depth_method = MethodSpecification(
    config=Difix3DTrainerConfig(
        method_name="difix3d-depth",
        experiment_name="difix3d-depth",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=Difix3DPipelineConfig(
            datamanager=Difix3DDataManagerConfig(
                dataparser=BlenderDataParserConfig(data=Path("data/materials")),
                train_num_rays_per_batch=16384,
                eval_num_rays_per_batch=4096,
            ),
            model=Difix3DModelConfig(
                eval_num_rays_per_chunk=32768,
                average_init_density=0.01,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            diffusion_ckpt_path=Path("outputs/materials/degraded/depth/checkpoints/model_2000.pkl"),
            use_depth_adapter=True,
            use_canny_adapter=False,
            steps_per_fix=2000,
        ),
        optimizers=common_optimizers,
        viewer=ViewerConfig(num_rays_per_chunk=32768),
        vis="viewer+wandb",
        logging=LoggingConfig(steps_per_log=10),
    ),
    description="Difix3D (Depth Adapter Only)",
)

difix3d_canny_method = MethodSpecification(
    config=Difix3DTrainerConfig(
        method_name="difix3d-canny",
        experiment_name="difix3d-canny",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=Difix3DPipelineConfig(
            datamanager=Difix3DDataManagerConfig(
                dataparser=BlenderDataParserConfig(data=Path("data/materials")),
                train_num_rays_per_batch=16384,
                eval_num_rays_per_batch=4096,
            ),
            model=Difix3DModelConfig(
                eval_num_rays_per_chunk=32768,
                average_init_density=0.01,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            diffusion_ckpt_path=Path("outputs/materials/degraded/canny/checkpoints/model_2000.pkl"),
            use_depth_adapter=False,
            use_canny_adapter=True,
            steps_per_fix=2000,
        ),
        optimizers=common_optimizers,
        viewer=ViewerConfig(num_rays_per_chunk=32768),
        vis="viewer+wandb",
        logging=LoggingConfig(steps_per_log=10),
    ),
    description="Difix3D (Canny Adapter Only)",
)

difix3d_depth_canny_method = MethodSpecification(
    config=Difix3DTrainerConfig(
        method_name="difix3d-depth-canny",
        experiment_name="difix3d-depth-canny",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=Difix3DPipelineConfig(
            datamanager=Difix3DDataManagerConfig(
                dataparser=BlenderDataParserConfig(data=Path("data/materials")),
                train_num_rays_per_batch=16384,
                eval_num_rays_per_batch=4096,
            ),
            model=Difix3DModelConfig(
                eval_num_rays_per_chunk=32768,
                average_init_density=0.01,
                camera_optimizer=CameraOptimizerConfig(mode="off"),
            ),
            diffusion_ckpt_path=Path("outputs/materials/degraded/depth_canny/checkpoints/model_2000.pkl"),
            use_depth_adapter=True,
            use_canny_adapter=True,
            steps_per_fix=2000,
        ),
        optimizers=common_optimizers,
        viewer=ViewerConfig(num_rays_per_chunk=32768),
        vis="viewer+wandb",
        logging=LoggingConfig(steps_per_log=10),
    ),
    description="Difix3D (Structure Guided - Full)",
)