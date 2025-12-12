# SAR Crop Assessment - Harmonic Analysis & Synthetic Data Generation

## 🌾 Project Overview

This project bridges **remote sensing** and **deep learning** to predict crop growth stages (sowing, maturity, harvesting) using Synthetic Aperture Radar (SAR) imagery. It combines Fourier harmonic analysis of temporal backscatter signals with CNN-based classification on synthetically augmented training data.

## 🎯 Why Fourier Series?

Crop growth follows seasonal patterns. SAR backscatter values collected at regular 15-day intervals show periodic behavior—low during sowing, rising through growth, peaking at maturity, then declining toward harvest.

**Fourier Series** decomposes this periodic signal into **harmonic components**:

```
S(t) = a₀/2 + Σ[aₙ cos(nωt) + bₙ sin(nωt)]  for n=1 to 16
```

Where:
- **a₀/2** = mean backscatter (DC component)
- **aₙ, bₙ** = harmonic coefficients (amplitudes of each frequency)
- **n** = harmonic order (1-16 in this project)
- **ω** = angular frequency of annual cycle

**Benefits:**
- Extracts periodic patterns from noisy SAR data
- Captures crop phenology as harmonic components
- Enables statistical significance testing (p < 0.05)
- Generates synthetic variations that preserve temporal characteristics

---

## 📊 Project Workflow Map

```
┌─────────────────────────────────────────────────────────────────┐
│                    SAR DATA ACQUISITION                         │
│  (Sentinel-1, Radarsat-2, etc. - 15-day intervals)              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│              TEMPORAL BACKSCATTER ANALYSIS                      │
│  Extract average σ⁰ (backscatter coefficient) per date          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│           FOURIER SERIES DECOMPOSITION (MATLAB)                 │
│  • Decompose S(t) into 16 harmonic terms                        │
│  • Extract aₙ, bₙ, amplitude, phase, frequency                   │
│  • Perform significance testing (p < 0.05)                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│           IMAGE SYNTHESIS (SNAP TOOL)                           │
│  • Use trapezoidal rule with significant coefficients           │
│  • Generate 10 SAR images per crop sample                       │
│  • Formulas: S(t) = Σ[aₙ cos(nωt) + bₙ sin(nωt)]                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│         SYNTHETIC DATA GENERATION (THIS PROJECT)                │
│  • Load 10 base SAR images per sample                           │
│  • Apply augmentation pipeline (speckle, distortion, etc.)      │
│  • Generate 200 synthetic sample sets                           │
│  • Output: 2000 total images (10 × 200)                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│              CNN MODEL TRAINING & CLASSIFICATION                │
│  Input: 10 temporal SAR images                                  │
│  Output: Crop stage (Sowing / Maturity / Harvesting)            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Processing Pipeline

```
Raw SAR Signal (Annual Cycle)
    ↓
[Sowing] ──→ [Germination] ──→ [Vegetative Growth] ──→ [Maturity] ──→ [Harvest]
    ↓              ↓                    ↓                   ↓             ↓
  σ⁰ = -15dB    σ⁰ = -10dB          σ⁰ = -8dB           σ⁰ = -5dB    σ⁰ = -12dB
  
     ↓
Collect at 15-day intervals (9 measurements/year)
     ↓
Apply Fourier Series Decomposition
     ↓
Extract Significant Harmonics (p < 0.05)
     ↓
Synthesize 10 Representative Images
     ↓
Augment → 200 Synthetic Samples
     ↓
Train CNN Classifier
```

---

## 📦 System Architecture

```
INPUT LAYER
    │
    ├─→ Sample_001/
    │   ├─ Image_Day_0.png (sowing)
    │   ├─ Image_Day_15.png
    │   ├─ ...
    │   └─ Image_Day_135.png (harvest)
    │
    ├─→ Sample_002/
    │   └─ [10 images]
    │
    └─→ Sample_200/
        └─ [10 images]

           ↓

AUGMENTATION LAYER
    ├─ Speckle Noise (SAR-specific)
    ├─ Brightness-Contrast Adjustment
    ├─ Elastic Distortion
    ├─ Grid Distortion
    └─ Gaussian Noise

           ↓

TRAINING DATASET
    ├─ 200 samples × 10 images = 2000 images
    ├─ Split: 70% train, 15% val, 15% test
    └─ Ready for CNN input

           ↓

CNN CLASSIFIER
    ├─ Conv Layers (feature extraction)
    ├─ Dense Layers (classification)
    └─ Output: [Sowing, Maturity, Harvesting]
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install albumentations==1.3.0 opencv-python pillow tqdm numpy
```

### Step 1: Prepare Base Images

Organize 10 SAR images (one per 15-day interval) in your project folder.

### Step 2: Generate Synthetic Data

```bash
git clone https://github.com/Prowin7/Harmonic_Analysis_CROP.git
cd Harmonic_Analysis_CROP
python scripts/generate_synthetic_data.py
```

### Step 3: Output

```
sar_synthetic_samples/
├── Sample_001/
│   ├── image_01.png
│   ├── image_02.png
│   └── ... (10 images)
├── Sample_002/
└── Sample_200/
```

---

## 🎨 Augmentation Pipeline

| Transformation | Probability | SAR Relevance |
|---|---|---|
| **Speckle Noise** | 100% | Multiplicative noise inherent to SAR |
| **Brightness-Contrast** | 70% | Sensor gain & radiometric variations |
| **Gamma Correction** | 70% | Atmospheric attenuation effects |
| **CLAHE** | 40% | Enhance local terrain features |
| **Gaussian Noise** | 60% | Thermal noise & quantization error |
| **Blur** | 30% | Resolution degradation over distance |
| **Elastic Distortion** | 25% | Registration errors & terrain undulation |
| **Grid Distortion** | 25% | Geometric distortions from relief |

---

## 📐 Mathematical Foundation

### Fourier Series Representation

Given temporal backscatter sequence S(t):

```
S(t) = a₀/2 + Σ[aₙ cos(nωt) + bₙ sin(nωt)]
       n=1 to 16

where:
  aₙ = (2/T) ∫₀ᵀ S(t) cos(nωt) dt
  bₙ = (2/T) ∫₀ᵀ S(t) sin(nωt) dt
  ω  = 2π/T (T = 1 year = 365 days)
```

### Amplitude & Phase

For each harmonic n:

```
Amplitude(n) = √(aₙ² + bₙ²)
Phase(n) = arctan(bₙ/aₙ)
```

### Significance Testing

- **Null Hypothesis:** Coefficient = 0
- **Test:** MATLAB significance testing routine
- **Threshold:** p < 0.05
- **Retention:** Only significant coefficients retained for synthesis

### Image Synthesis (Trapezoidal Rule)

```
S_synthetic(tᵢ) = a₀/2 + Σ[aₙ cos(nωtᵢ) + bₙ sin(nωtᵢ)]
                  n∈significant
```

Convert S_synthetic(tᵢ) to image intensities using SNAP tool.

---

## 📂 Project Structure

```
Harmonic_Analysis_CROP/
├── README.md                          # Project documentation
├── requirements.txt                   # Dependencies
│
├── scripts/
│   ├── generate_synthetic_data.py     # Main augmentation script
│   ├── train_model.py                 # CNN training
│   └── evaluate_model.py              # Model evaluation
│
├── data/
│   ├── raw/                           # Original 10 SAR images
│   └── synthetic/                     # Generated 200 sample folders
│
├── models/
│   └── crop_classifier.h5             # Trained CNN weights
│
└── notebooks/
    └── fourier_analysis.ipynb         # Fourier decomposition demo
```

---

## 💾 Data Input Format

**Source:** SAR backscatter images synthesized from Fourier coefficients

**Format:** Grayscale PNG (8-bit, 256×256 pixels recommended)

**Temporal Coverage:** 
- Day 0: Sowing
- Day 15-135: Growth monitoring
- 10 images spanning ~135 days

**Organization:**

```
base_images/
├── img_day_0.png
├── img_day_15.png
├── img_day_30.png
├── ...
└── img_day_135.png
```

---

## ✅ Key Metrics

| Metric | Value | Purpose |
|---|---|---|
| Base Images | 10 | Temporal sequence per crop |
| Synthetic Samples | 200 | Adequate CNN training set |
| Total Generated Images | 2,000 | 200 samples × 10 images |
| Fourier Terms | 16 | Captures up to 8 harmonics |
| Significance Level | p < 0.05 | Statistical rigor |
| Image Resolution | 256×256 | Standard SAR input |
| Augmentation Diversity | 8 techniques | Robust model training |

---

## 🔍 Usage Examples

### Generate Dataset
```python
python scripts/generate_synthetic_data.py --num-samples 200 --output-dir ./data/synthetic
```

### Train Classifier
```python
python scripts/train_model.py --data-path ./data/synthetic --epochs 50 --batch-size 32
```

### Evaluate Model
```python
python scripts/evaluate_model.py --model ./models/crop_classifier.h5 --test-data ./data/synthetic
```

---

## 📈 Expected Results

- **Dataset Size:** 200 folders, 2000 images (~200-500 MB)
- **Training Time:** ~30-60 minutes (GPU recommended)
- **Expected Accuracy:** 85-95% (depends on base image quality)
- **Model Inference:** ~0.1-0.3 sec per 10-image sequence

---

## 🛠️ Customization

**To modify augmentation intensity:**
```python
# In generate_synthetic_data.py
augment = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.6),  # Increase noise variance
    # ... other augmentations
])
```

**To change number of samples:**
```python
NUM_SAMPLES = 500  # Generate 500 instead of 200
```

**To adjust speckle noise:**
```python
def add_speckle(img, var=0.20):  # Increase from 0.10 to 0.20
```

---

## 📚 References

- Backscatter Analysis for Crop Phenology Monitoring
- Fourier Series Methods for Temporal SAR Analysis
- Synthetic Data Augmentation for Remote Sensing
- CNN Architectures for Agricultural Classification
- SNAP (Sentinel Application Platform) Documentation

---

## 📝 License

[Specify your license - MIT, Apache 2.0, etc.]

---

## 👤 Author

**Project:** Harmonic Analysis of Crop Growth using SAR Data  
**Repository:** https://github.com/Prowin7/Harmonic_Analysis_CROP.git  
**Contact:** [Your contact information]

---

## 🎓 Citation

If you use this project in research, please cite:

```
@software{harmonic_analysis_crop_2024,
  author={Your Name},
  title={Harmonic Analysis & Synthetic Data Generation for SAR-based Crop Assessment},
  year={2024},
  url={https://github.com/Prowin7/Harmonic_Analysis_CROP}
}
```
