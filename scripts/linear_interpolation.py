import argparse
import os
import cv2

def is_image_file(filename):
    ext = filename.lower().split(".")[-1]
    return ext in ["png", "jpg", "jpeg", "bmp", "tiff", "webp"]

def resize_images(root_dir, target_width=480, target_height=270):
    # Iterate over all hash directories in the root
    for hash_dir in os.listdir(root_dir):
        hash_path = os.path.join(root_dir, hash_dir)
        if not os.path.isdir(hash_path):
            continue

        images_dir = os.path.join(hash_path, "nerfstudio", "images_8")
        if not os.path.isdir(images_dir):
            print(f"Skipping, images_8 not found in: {hash_path}")
            continue

        # Iterate over all images in images_8
        for filename in os.listdir(images_dir):
            if not is_image_file(filename):
                continue

            img_path = os.path.join(images_dir, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Skipping (not readable): {img_path}")
                continue

            # Resize image
            resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(img_path, resized)
            print(f"Resized → {img_path}")

    print("\nDone! All images in images_8 directories resized.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all images in images_8 folders of hash directories to 270x480.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the root directory containing hash subdirectories.")
    args = parser.parse_args()

    resize_images(args.input_dir)
