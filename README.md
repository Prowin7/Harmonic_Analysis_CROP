# SAR Crop Assessment - Synthetic Data Generation

## Overview

This project generates synthetic SAR (Synthetic Aperture Radar) image samples for crop growth stage classification. It uses advanced data augmentation techniques to create 200 diverse training samples from 10 base SAR images, enabling robust CNN model training for crop assessment tasks.

## Workflow

```
Raw SAR Images (10 per crop)
        ↓
  Augmentation Pipeline
        ↓
  200 Synthetic Samples
        ↓
  CNN Model Training
        ↓
Crop Stage Classification (Sowing, Maturity, Harvesting)
```

## Motivation

SAR imagery captures crop backscatter patterns at regular 15-day intervals. Fourier series decomposition extracts significant spectral components. Limited base images necessitate synthetic data generation to train robust deep learning models for crop phenology prediction.

## Features

- **SAR-Specific Augmentation:** Multiplicative speckle noise mimics SAR acquisition characteristics
- **Advanced Transformations:** Elastic and grid distortions, brightness/contrast adjustments
- **Scalable Generation:** Create 200+ samples from minimal base data
- **Batch Processing:** Efficient tqdm-based progress tracking
- **Auto-Download:** Generated dataset zipped and ready for local use

## Requirements

```
albumentations==1.3.0
opencv-python
pillow
tqdm
numpy
```

Install dependencies:
```bash
pip install albumentations==1.3.0 opencv-python pillow tqdm
```

## Usage

### Step 1: Upload Base Images

Upload 10 SAR images (one per 15-day interval) when prompted. Images should be grayscale PNG/JPG format.

### Step 2: Run Augmentation Script

```python
!pip install albumentations==1.3.0 opencv-python pillow tqdm

import cv2
import os
from google.colab import files
import albumentations as A
from tqdm import tqdm
from PIL import Image
import numpy as np

# Upload images (10 per sample)
uploaded = files.upload()
input_images = list(uploaded.keys())

# Output directory
BASE_OUTPUT = "/content/sar_synthetic_samples"
os.makedirs(BASE_OUTPUT, exist_ok=True)

# SAR speckle noise (multiplicative)
def add_speckle(img, var=0.10):
    img_f = img.astype(np.float32) / 255.0
    noise = np.random.normal(0, var, img_f.shape).astype(np.float32)
    out = img_f + img_f * noise
    return np.clip(out * 255, 0, 255).astype(np.uint8)

# Augmentation pipeline
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

# Generate 200 synthetic samples
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
        out.save(os.path.join(sample_folder, img_name))

print("✓ Done — synthetic dataset generated successfully!")

# Download as ZIP
!zip -r "/content/sar_200_samples.zip" "/content/sar_synthetic_samples" > /dev/null
files.download("/content/sar_200_samples.zip")
```

### Step 3: Output Structure

```
sar_synthetic_samples/
├── Sample_001/
│   ├── image_01.png
│   ├── image_02.png
│   └── ...
│   └── image_10.png
├── Sample_002/
│   └── ...
└── Sample_200/
    └── ...
```

Each sample folder contains 10 augmented SAR images representing the 15-day temporal sequence.

## Augmentation Pipeline

| Augmentation | Probability | Purpose |
|---|---|---|
| Brightness-Contrast | 70% | Simulate sensor gain variations |
| Gamma Correction | 70% | Handle radiometric distortions |
| CLAHE | 40% | Enhance local contrast |
| Gaussian Noise | 60% | Add thermal noise |
| Blur | 30% | Simulate resolution loss |
| Elastic Distortion | 25% | Geometric registration errors |
| Grid Distortion | 25% | Terrain-induced distortions |
| Speckle Noise | 100% | SAR-specific multiplicative noise |

## Technical Details

### Speckle Noise Function
SAR images contain multiplicative speckle noise. The implementation adds Gaussian noise scaled by image intensity:

```python
noise = N(0, σ²)
output = input + input × noise
```

where σ = 0.10 (10% variance)

### Data Flow

1. **Input:** Grayscale image (256×256 or custom size)
2. **Augmentation:** Applied sequentially using Albumentations
3. **Speckle Addition:** Multiplicative noise injection
4. **Output:** Augmented grayscale image (8-bit PNG)

## Output

- **Dataset Size:** 200 folders × 10 images = 2000 total images
- **File Format:** PNG (grayscale, 8-bit)
- **Download:** `sar_200_samples.zip` (~[size depends on image resolution])

## Next Steps

1. Extract downloaded ZIP file
2. Organize samples with crop stage labels (sowing, maturity, harvesting)
3. Train CNN model for classification:
   ```python
   # Load samples
   # Define CNN architecture
   # Train with 200 synthetic samples
   # Evaluate on validation set
   ```

## Parameters to Adjust

- **NUM_SAMPLES:** Change 200 to desired number of synthetic samples
- **Speckle variance:** Modify `var=0.10` in `add_speckle()`
- **Augmentation probabilities:** Tune individual `p=` values
- **Image size:** Process images of any resolution

## Notes

- Augmentations are deterministic per run (set random seed for reproducibility)
- Each sample folder maintains the same 10 image names as originals
- Speckle noise varies per image, creating synthetic diversity
- CLAHE parameters tuned for 256×256 images; adjust `tile_grid_size` for larger images

## Performance

- Generation time: ~2-5 minutes for 200 samples (varies with image size/resolution)
- Storage: Depends on image dimensions (typically 100-500 MB for full dataset)

## License

[Add your license here]

## Author

[Praveen Nukilla/IIITA]