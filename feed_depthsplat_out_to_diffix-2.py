import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from Difix3D.src.pipeline_difix import DifixPipeline
from diffusers.utils import load_image

def iter_image_files(root: Path):
    exts = {".png", ".jpg", ".jpeg"}
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if Path(fname).suffix.lower() in exts:
                yield Path(dirpath) / fname

def to_numpy(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img).astype("float32")
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    return arr / 255.0

def to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr * 255.0, 0, 255).astype("uint8")
    return Image.fromarray(arr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prev_input_dir", default=None)
    parser.add_argument("--lambda_prior", type=float, default=0.0)
    args = parser.parse_args()

    input_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    prev_root = Path(args.prev_input_dir).resolve() if args.prev_input_dir else None
    output_root.mkdir(parents=True, exist_ok=True)

    difix_pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)

    for in_path in iter_image_files(input_root):
        rel = in_path.relative_to(input_root)
        out_path = output_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        depth_img = load_image(str(in_path))
        result = difix_pipe(prompt="remove degradation", image=depth_img).images[0]

        if args.lambda_prior > 0 and prev_root is not None:
            prev_path = prev_root / rel
            if prev_path.exists():
                prev_img = load_image(str(prev_path)).resize(result.size)
                prev_np = to_numpy(prev_img)
                diffix_np = to_numpy(result)
                lam = args.lambda_prior
                blended = (1-lam)*prev_np + lam*diffix_np
                to_pil(blended).save(out_path)
            else:
                result.save(out_path)
        else:
            result.save(out_path)

if __name__ == "__main__":
    main()
