import json
import torch
import random
import numpy as np
import cv2
from PIL import Image, ImageFilter
import torchvision.transforms.functional as F
from torchvision import transforms

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, height=512, width=512, tokenizer=None, degrade_inputs=False, load_canny=False):
        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]
        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer
        self.degrade_inputs = degrade_inputs
        self.load_canny = load_canny # New flag

    def __len__(self):
        return len(self.img_ids)

    def simulate_artifacts(self, img_pil):
        """
        Corrupts the input image to simulate NeRF artifacts/blur.
        """
        if random.random() < 0.8: # 80% chance to blur
            radius = random.uniform(1, 3)
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius))
            
        if random.random() < 0.5: # 50% chance to resize
            w, h = img_pil.size
            factor = random.uniform(2, 4)
            small_w, small_h = int(w/factor), int(h/factor)
            img_pil = img_pil.resize((small_w, small_h), resample=Image.BILINEAR)
            img_pil = img_pil.resize((w, h), resample=Image.NEAREST)
            
        return img_pil

    def get_canny_edge(self, image_pil):
        # Convert to numpy
        image_np = np.array(image_pil)
        # Detect edges
        low_threshold = 100
        high_threshold = 200
        edges = cv2.Canny(image_np, low_threshold, high_threshold)
        # Convert back to PIL L mode
        edges_pil = Image.fromarray(edges).convert("L")
        return edges_pil

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        input_path = self.data[img_id]["image"]
        target_path = self.data[img_id]["target_image"]
        depth_path = self.data[img_id].get("depth_image", None)
        caption = self.data[img_id]["prompt"]
        
        try:
            input_pil = Image.open(input_path).convert("RGB")
            target_pil = Image.open(target_path).convert("RGB")
            depth_pil = Image.open(depth_path).convert("L") if depth_path else None
            
            if self.degrade_inputs:
                input_pil = self.simulate_artifacts(input_pil)

            # Compute Canny on the fly from the input (or target, depending on task logic)
            # Typically for control, we want the structure of the input
            canny_pil = None
            if self.load_canny:
                # We extract edges from the INPUT image (which might be degraded) 
                # or the clean TARGET structure depending on training objective.
                # Assuming we want to guide restoration based on input structure:
                canny_pil = self.get_canny_edge(input_pil)

        except Exception as e:
            print(f"Error loading {img_id}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        input_t = F.to_tensor(input_pil)
        input_t = F.resize(input_t, self.image_size, interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        input_t = F.normalize(input_t, mean=[0.5], std=[0.5])

        target_t = F.to_tensor(target_pil)
        target_t = F.resize(target_t, self.image_size, interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        target_t = F.normalize(target_t, mean=[0.5], std=[0.5])

        # Depth Processing
        if depth_pil is not None:
            depth_t = F.to_tensor(depth_pil)
            depth_t = F.resize(depth_t, self.image_size, interpolation=F.InterpolationMode.NEAREST)
        else:
            depth_t = torch.zeros((1, self.image_size[0], self.image_size[1]))

        # Canny Processing
        if canny_pil is not None:
            canny_t = F.to_tensor(canny_pil)
            canny_t = F.resize(canny_t, self.image_size, interpolation=F.InterpolationMode.NEAREST)
        else:
            canny_t = torch.zeros((1, self.image_size[0], self.image_size[1]))

        out = {
            "output_pixel_values": target_t.unsqueeze(0), 
            "conditioning_pixel_values": input_t.unsqueeze(0), 
            "depth_pixel_values": depth_t,
            "canny_pixel_values": canny_t,
            "caption": caption,
        }
        
        if self.tokenizer:
            out["input_ids"] = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids

        return out