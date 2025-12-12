!pip install albumentations==1.3.0 opencv-python pillow tqdm

import cv2
import os
from google.colab import files
import albumentations as A
from tqdm import tqdm
from PIL import Image
import numpy as np

# -------- Upload images (10 per sample) --------
uploaded = files.upload()
input_images = list(uploaded.keys())

# -------- Output base directory --------
BASE_OUTPUT = "/content/sar_synthetic_samples"
os.makedirs(BASE_OUTPUT, exist_ok=True)

# -------- Speckle noise multiplicative --------
def add_speckle(img, var=0.10):
    img_f = img.astype(np.float32) / 255.0
    noise = np.random.normal(0, var, img_f.shape).astype(np.float32)
    out = img_f + img_f * noise
    return np.clip(out * 255, 0, 255).astype(np.uint8)

# -------- Augmentation pipeline --------
augment = A.Compose([
    A.RandomBrightnessContrast(p=0.7),
    A.RandomGamma(p=0.7),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.4),
    A.GaussNoise(var_limit=(5.0, 40.0), p=0.6),
    A.Blur(blur_limit=3, p=0.3),
    A.ElasticTransform(alpha=12, sigma=4, alpha_affine=4,
                       border_mode=cv2.BORDER_REFLECT_101, p=0.25),
    A.GridDistortion(num_steps=4, distort_limit=0.06,
                     border_mode=cv2.BORDER_REFLECT_101, p=0.25)
])

# -------- Number of synthetic samples you want --------
NUM_SAMPLES = 200

print("Generating 200 synthetic sample folders...")

for sample_id in tqdm(range(1, NUM_SAMPLES + 1)):
    sample_folder = os.path.join(BASE_OUTPUT, f"Sample_{sample_id:03d}")
    os.makedirs(sample_folder, exist_ok=True)

    for img_name in input_images:
        pil_img = Image.open(img_name).convert("L")
        img_np = np.array(pil_img)

        aug_img = augment(image=img_np)["image"]
        aug_img = add_speckle(aug_img)

        out = Image.fromarray(aug_img, mode="L")

        # same filename saved inside every folder
        out.save(os.path.join(sample_folder, img_name))

print("✓ Done — synthetic dataset generated successfully!")

# -------- Download the generated dataset as a zip file --------
!zip -r "/content/sar_200_samples.zip" "/content/sar_synthetic_samples" > /dev/null
files.download("/content/sar_200_samples.zip")
