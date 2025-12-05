import os
import gc
import lpips
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import transformers
from torchvision.transforms.functional import crop
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from glob import glob
from einops import rearrange

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb

from model import Difix, load_ckpt_from_state_dict, save_ckpt
from dataset import PairedDataset
from loss import gram_loss


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    # DYNAMIC INITIALIZATION BASED ON ARGUMENT
    net_difix = Difix(
        lora_rank_vae=args.lora_rank_vae, 
        timestep=args.timestep, 
        mv_unet=args.mv_unet,
        use_depth_adapter=args.use_depth,
        use_canny_adapter=args.use_canny,
    )
    net_difix.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_difix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_difix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)
    
    net_vgg = torchvision.models.vgg16(weights="DEFAULT").features
    for param in net_vgg.parameters():
        param.requires_grad_(False)

    # Optimizer setup
    layers_to_opt = []
    
    # 1. Depth Adapter
    if net_difix.adapter is not None:
        print("Training WITH Depth Adapter parameters.")
        layers_to_opt += list(net_difix.adapter.parameters())
    
    # 2. Canny Adapter
    if net_difix.adapter_canny is not None:
        print("Training WITH Canny Adapter parameters.")
        layers_to_opt += list(net_difix.adapter_canny.parameters())

    # 3. VAE LoRA (Always trained)
    for n, _p in net_difix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
            
    # 4. VAE Skip Convs (Always trained)
    layers_to_opt = layers_to_opt + list(net_difix.vae.decoder.skip_conv_1.parameters()) + \
        list(net_difix.vae.decoder.skip_conv_2.parameters()) + \
        list(net_difix.vae.decoder.skip_conv_3.parameters()) + \
        list(net_difix.vae.decoder.skip_conv_4.parameters())

    optimizer = torch.optim.AdamW(list(set(layers_to_opt)), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)
    
    if os.path.isdir(args.dataset_path):
        dataset_name = os.path.basename(os.path.normpath(args.dataset_path))
        args.dataset_path = os.path.join(args.dataset_path, f"{dataset_name}.json")

    dataset_train = PairedDataset(
        dataset_path=args.dataset_path, 
        split="train", 
        tokenizer=net_difix.tokenizer,
        degrade_inputs=args.degrade_inputs,
        load_canny=args.use_canny
    )
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dataset_val = PairedDataset(dataset_path=args.dataset_path, split="test", tokenizer=net_difix.tokenizer)
    random.Random(42).shuffle(dataset_val.img_ids)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # Resume logic
    global_step = 0     
    if args.resume is not None:
        if os.path.isdir(args.resume):
            ckpt_files = glob(os.path.join(args.resume, "*.pkl"))
            assert len(ckpt_files) > 0, f"No checkpoint files found: {args.resume}"
            ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split("/")[-1].replace("model_", "").replace(".pkl", "")))
            print("="*50); print(f"Loading checkpoint from {ckpt_files[-1]}"); print("="*50)
            global_step = int(ckpt_files[-1].split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_difix, optimizer = load_ckpt_from_state_dict(
                net_difix, optimizer, ckpt_files[-1]
            )
        elif args.resume.endswith(".pkl"):
            print("="*50); print(f"Loading checkpoint from {args.resume}"); print("="*50)
            global_step = int(args.resume.split("/")[-1].replace("model_", "").replace(".pkl", ""))
            net_difix, optimizer = load_ckpt_from_state_dict(
                net_difix, optimizer, args.resume
            )     
        else:
            raise NotImplementedError(f"Invalid resume path: {args.resume}")
    else:
        print("="*50); print(f"Training from scratch"); print("="*50)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move to device
    net_difix.to(accelerator.device, dtype=weight_dtype)
    if net_difix.adapter: 
        net_difix.adapter.to(accelerator.device, dtype=weight_dtype)
    if net_difix.adapter_canny:
        net_difix.adapter_canny.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_vgg.to(accelerator.device, dtype=weight_dtype)
    
    # Prepare
    net_difix, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_difix, optimizer, dl_train, lr_scheduler
    )
    net_lpips, net_vgg = accelerator.prepare(net_lpips, net_vgg)
    t_vgg_renorm =  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if accelerator.is_main_process:
        # 1. Clean Dataset Name
        dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
        
        # 2. Clean Mode String
        mode_str = "degraded" if args.degrade_inputs else "clean"
        
        # 3. Create a Single "Model Variant" String (The missing piece for easy filtering)
        if args.use_depth and args.use_canny:
            model_variant = "both_adapters"
        elif args.use_depth:
            model_variant = "depth_adapter"
        elif args.use_canny:
            model_variant = "canny_adapter"
        else:
            model_variant = "base_model"

        # 4. Construct Group Name
        # Grouping by variant + dataset allows comparing clean vs degraded lines on one chart
        group_name = f"{dataset_name}-{model_variant}"

        init_kwargs = {
            "wandb": {
                "name": args.tracker_run_name, 
                "dir": args.output_dir,
                "group": group_name,
                "job_type": "training",
                "tags": [dataset_name, mode_str, model_variant] 
            }
        }        
        
        # 5. Add these clean strings to the config so you can "Group By" them
        tracker_config = dict(vars(args))
        tracker_config["dataset_name"] = dataset_name
        tracker_config["mode_str"] = mode_str 
        tracker_config["model_variant"] = model_variant
        
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps", disable=not accelerator.is_local_main_process)

    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_difix]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"].to(weight_dtype)
                x_tgt = batch["output_pixel_values"].to(weight_dtype)
                
                # Load maps if present
                depth_map = batch["depth_pixel_values"].to(weight_dtype) if "depth_pixel_values" in batch else None
                canny_map = batch["canny_pixel_values"].to(weight_dtype) if "canny_pixel_values" in batch else None
                
                B, V, C, H, W = x_src.shape

                # forward pass
                x_tgt_pred = net_difix(
                    x_src, 
                    prompt_tokens=batch["input_ids"], 
                    depth_map=depth_map,
                    canny_map=canny_map
                )
                
                x_tgt = rearrange(x_tgt, 'b v c h w -> (b v) c h w')
                x_tgt_pred = rearrange(x_tgt_pred, 'b v c h w -> (b v) c h w')
                          
                # Losses
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips
                
                if args.lambda_gram > 0:
                    if global_step > args.gram_loss_warmup_steps:
                        x_tgt_pred_renorm = t_vgg_renorm(x_tgt_pred * 0.5 + 0.5)
                        crop_h, crop_w = 400, 400
                        top, left = random.randint(0, H - crop_h), random.randint(0, W - crop_w)
                        x_tgt_pred_renorm = crop(x_tgt_pred_renorm, top, left, crop_h, crop_w)
                        x_tgt_renorm = t_vgg_renorm(x_tgt * 0.5 + 0.5)
                        x_tgt_renorm = crop(x_tgt_renorm, top, left, crop_h, crop_w)
                        loss_gram = gram_loss(x_tgt_pred_renorm.to(weight_dtype), x_tgt_renorm.to(weight_dtype), net_vgg) * args.lambda_gram
                        loss += loss_gram
                    else:
                        loss_gram = torch.tensor(0.0).to(weight_dtype)                     

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                
                x_tgt = rearrange(x_tgt, '(b v) c h w -> b v c h w', v=V)
                x_tgt_pred = rearrange(x_tgt_pred, '(b v) c h w -> b v c h w', v=V)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    if args.lambda_gram > 0:
                        logs["loss_gram"] = loss_gram.detach().item()
                    progress_bar.set_postfix(**logs)

                    # viz some images
                    if global_step % args.viz_freq == 1:
                        log_dict = {
                            "train/source": [wandb.Image(rearrange(x_src, "b v c h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/target": [wandb.Image(rearrange(x_tgt, "b v c h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/model_output": [wandb.Image(rearrange(x_tgt_pred, "b v c h w -> b c (v h) w")[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                        }
                        
                        if depth_map is not None:
                            log_dict["train/depth"] = [wandb.Image(depth_map[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)]
                        if canny_map is not None:
                            log_dict["train/canny"] = [wandb.Image(canny_map[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)]

                        for k in log_dict:
                            logs[k] = log_dict[k]

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1 or global_step == args.max_train_steps:
                        ckpt_dir = os.path.join(args.output_dir, "checkpoints")
                        outf = os.path.join(ckpt_dir, f"model_{global_step}.pkl")
                        
                        save_ckpt(accelerator.unwrap_model(net_difix), optimizer, outf)
                        
                        all_ckpts = glob(os.path.join(ckpt_dir, "model_*.pkl"))
                        
                        for ckpt in all_ckpts:
                            if ckpt == outf:
                                continue
                            try:
                                os.remove(ckpt)
                                if accelerator.is_main_process:
                                    print(f"Removed old checkpoint: {ckpt}")
                            except OSError as e:
                                print(f"Error deleting checkpoint {ckpt}: {e}")

                    # compute validation set L2, LPIPS
                    if args.eval_freq > 0 and global_step % args.eval_freq == 1:
                        l_l2, l_lpips = [], []
                        log_dict = {"sample/source": [], "sample/target": [], "sample/model_output": []}
                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break
                            x_src = batch_val["conditioning_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            x_tgt = batch_val["output_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                            
                            # Handle depth/canny map for validation
                            val_depth = batch_val["depth_pixel_values"].to(accelerator.device, dtype=weight_dtype) if "depth_pixel_values" in batch_val else None
                            val_canny = batch_val["canny_pixel_values"].to(accelerator.device, dtype=weight_dtype) if "canny_pixel_values" in batch_val else None
                            
                            B, V, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # forward pass
                                x_tgt_pred = accelerator.unwrap_model(net_difix)(
                                    x_src, 
                                    prompt_tokens=batch_val["input_ids"].cuda(), 
                                    depth_map=val_depth,
                                    canny_map=val_canny
                                )
                                
                                if step % 10 == 0:
                                    log_dict["sample/source"].append(wandb.Image(rearrange(x_src, "b v c h w -> b c (v h) w")[0].float().detach().cpu(), caption=f"idx={len(log_dict['sample/source'])}"))
                                    log_dict["sample/target"].append(wandb.Image(rearrange(x_tgt, "b v c h w -> b c (v h) w")[0].float().detach().cpu(), caption=f"idx={len(log_dict['sample/source'])}"))
                                    log_dict["sample/model_output"].append(wandb.Image(rearrange(x_tgt_pred, "b v c h w -> b c (v h) w")[0].float().detach().cpu(), caption=f"idx={len(log_dict['sample/source'])}"))
                                
                                x_tgt = x_tgt[:, 0] # take the input view
                                x_tgt_pred = x_tgt_pred[:, 0] # take the input view
                                # compute the reconstruction losses
                                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean")
                                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean()

                                l_l2.append(loss_l2.item())
                                l_lpips.append(loss_lpips.item())

                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
                        for k in log_dict:
                            logs[k] = log_dict[k]
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    accelerator.log(logs, step=global_step)

                # FIXED: Stop training exactly when max steps are reached
                if global_step >= args.max_train_steps:
                    break

        # FIXED: Break outer epoch loop as well
        if global_step >= args.max_train_steps:
            break


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_gram", default=1.0, type=float)
    parser.add_argument("--gram_loss_warmup_steps", default=2000, type=int)

    # dataset options
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--prompt", default=None, type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="difix", help="The name of the wandb project to log to.")
    parser.add_argument("--tracker_run_name", type=str, required=True)

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_vae", default=4, type=int)
    parser.add_argument("--timestep", default=199, type=int)
    parser.add_argument("--mv_unet", action="store_true")
    
    # New Flag: Use Depth Adapter
    parser.add_argument("--use_depth", action="store_true", help="Enable T2I Adapter for depth control")
    parser.add_argument("--use_canny", action="store_true", help="Enable T2I Adapter for Canny edge control")
    parser.add_argument("--degrade_inputs", action="store_true", help="Apply blur/downscaling to input images during training")

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=10_000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)
    
    # resume
    parser.add_argument("--resume", default=None, type=str)

    args = parser.parse_args()

    main(args)