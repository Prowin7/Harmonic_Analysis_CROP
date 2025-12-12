# ==========================================
# 1. SETUP & INSTALLATION
# ==========================================
!pip install torch torchvision torchaudio tqdm scikit-learn pandas pillow opencv-python --quiet

import os
import shutil
import zipfile
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from google.colab import drive
import random

# ==========================================
# 2. MOUNT DRIVE & EXTRACT DATA
# ==========================================
print("Mounting Google Drive...")
drive.mount('/content/drive')

# --- CONFIGURATION ---
DRIVE_ZIP_PATH = '/content/drive/MyDrive/dataset_1.zip'
DRIVE_LABEL_PATH = '/content/drive/MyDrive/labels.csv'

# Destination paths (Local Colab storage for speed)
DEST_DIR = "/content/dataset"
DATA = "/content/dataset/dataset_1"
LABEL_FILE = "/content/dataset/dataset_1/labels.csv"

print(f"\n[Step 1] Setting up data in {DEST_DIR}...")
if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

# Copy Labels
if os.path.exists(DRIVE_LABEL_PATH):
    if not os.path.exists(DATA): os.makedirs(DATA)
    shutil.copy(DRIVE_LABEL_PATH, LABEL_FILE)
    print("✅ Labels copied.")
else:
    print(f"❌ Error: labels.csv not found at {DRIVE_LABEL_PATH}")

# Extract Images
if os.path.exists(DRIVE_ZIP_PATH):
    print("Extracting images...")
    with zipfile.ZipFile(DRIVE_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DEST_DIR)
    print("✅ Extraction complete.")
else:
    print(f"❌ Error: dataset_1.zip not found at {DRIVE_ZIP_PATH}")

# ==========================================
# 3. DATA LOADING & STATISTICS
# ==========================================
labels = pd.read_csv(LABEL_FILE).set_index("sample_id")
BANDS = ["A1","A2","AMP1","AMP2","B1","B2","PHASE1","PHASE2","VAR1","VAR2"]

def load_sample(sample_id):
    """Loads 10-band stack for a sample."""
    folder = os.path.join(DATA, sample_id)
    imgs = []

    # Check for first band
    p_base = os.path.join(folder, f"{BANDS[0]}.png")
    if not os.path.exists(p_base):
        return np.zeros((32,32,10), dtype=np.float32)

    # Read base for dimensions
    base = cv2.imread(p_base, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    H, W = base.shape
    if base.max() > 1.5: base /= 255.0
    imgs.append(base)

    # Read other bands
    for b in BANDS[1:]:
        p = os.path.join(folder, f"{b}.png")
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            if img.shape != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
            if img.max() > 1.5: img /= 255.0
            imgs.append(img)
        else:
            imgs.append(np.zeros((H, W), dtype=np.float32))

    return np.stack(imgs, axis=-1)

# --- CALCULATE STATISTICS ---
print("\n[Step 2] Calculating precise dataset statistics...")
sample_ids = labels.index.tolist()
sample_ids = [sid for sid in sample_ids if os.path.exists(os.path.join(DATA, sid))]

sum_x = np.zeros(len(BANDS))
sum_x2 = np.zeros(len(BANDS))
total_pixel_count = 0

for sid in sample_ids:
    try:
        img_stack = load_sample(sid)
        flat_pixels = img_stack.reshape(-1, len(BANDS))
        sum_x += flat_pixels.sum(axis=0)
        sum_x2 += (flat_pixels ** 2).sum(axis=0)
        total_pixel_count += flat_pixels.shape[0]
    except: pass

band_mean = sum_x / max(total_pixel_count, 1)
band_var = (sum_x2 / max(total_pixel_count, 1)) - (band_mean ** 2)
band_std = np.sqrt(np.maximum(band_var, 0)) + 1e-8

print("✅ Band Mean:", np.round(band_mean, 4))
print("✅ Band Std: ", np.round(band_std, 4))

# ==========================================
# 4. DATASET & DATALOADERS
# ==========================================
PATCH = 32
STRIDE = 32

def extract_patches(sample):
    H, W, C = sample.shape
    if H < PATCH or W < PATCH:
        sample = cv2.resize(sample, (max(H, PATCH), max(W, PATCH)))
        H, W, C = sample.shape
    patches = []
    for y in range(0, H-PATCH+1, STRIDE):
        for x in range(0, W-PATCH+1, STRIDE):
            patches.append(sample[y:y+PATCH, x:x+PATCH, :])
    return np.array(patches)

class SARPhenologyDataset(Dataset):
    def __init__(self, sample_ids):
        self.data, self.targets = [], []
        for sid in sample_ids:
            arr = load_sample(sid)
            arr = (arr - band_mean) / (band_std + 1e-8)
            patches = extract_patches(arr)
            if len(patches) > 0:
                target = labels.loc[sid].to_numpy(dtype=np.float32)
                for p in patches:
                    self.data.append(p)
                    self.targets.append(target)
        if len(self.data) > 0:
            self.data = np.stack(self.data)
            self.targets = np.stack(self.targets)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx]).permute(2,0,1).float()
        y = torch.from_numpy(self.targets[idx]).float()
        return x, y

train_ids, val_ids = train_test_split(sample_ids, test_size=0.2, random_state=42)
print(f"\n[Step 3] Building Datasets (Train: {len(train_ids)}, Val: {len(val_ids)})...")

train_ds = SARPhenologyDataset(train_ids)
val_ds = SARPhenologyDataset(val_ids)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# ==========================================
# 5. MODEL DEFINITION
# ==========================================
class PhenologyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(10, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PhenologyCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# CHANGE: Use MSE for training (better convergence), track MAE for reporting
criterion = nn.MSELoss() 

# ==========================================
# 6. TRAINING LOOP (With Explicit MAE Print)
# ==========================================
def run_epoch(loader, training=True):
    model.train() if training else model.eval()
    total_loss = 0
    total_mae = 0
    n = 0
    
    if len(loader) == 0: return 0, 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        if training: optimizer.zero_grad()
        
        pred = model(x)
        loss = criterion(pred, y)      # MSE Loss for optimization
        mae = F.l1_loss(pred, y)       # MAE for human reporting
        
        if training:
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item() * x.size(0)
        total_mae += mae.item() * x.size(0)
        n += x.size(0)
        
    return total_loss / max(n, 1), total_mae / max(n, 1)

print("\n[Step 4] Starting Training...")
EPOCHS = 30
best_mae = float('inf')

for e in range(1, EPOCHS+1):
    # Get both Loss (MSE) and MAE
    train_loss, train_mae = run_epoch(train_loader, True)
    val_loss, val_mae = run_epoch(val_loader, False)
    
    print(f"Epoch {e:02d} | Loss: {train_loss:.2f} | Train MAE: {train_mae:.3f} | Val MAE: {val_mae:.3f}")
    
    if val_mae < best_mae:
        best_mae = val_mae
        torch.save(model.state_dict(), "best_model.pt")

print("✅ Training Complete. Best model saved.")

# ==========================================
# 7. PREDICTION & RESULTS
# ==========================================
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

def predict_sample(sample_id):
    try:
        arr = load_sample(sample_id)
        arr = (arr - band_mean) / (band_std + 1e-8)
        patches = extract_patches(arr)
        if len(patches) == 0: return np.zeros(3)
        
        t_patches = torch.from_numpy(patches).permute(0,3,1,2).float().to(device)
        preds = []
        with torch.no_grad():
            for i in range(0, len(t_patches), 64):
                preds.append(model(t_patches[i:i+64]).cpu().numpy())
        return np.mean(np.vstack(preds), axis=0)
    except: return np.zeros(3)

print("\n[Step 5] Generating Results...")
results = []
for sid in sample_ids:
    actual = labels.loc[sid].to_numpy(dtype=float)
    pred = predict_sample(sid)
    mae_err = np.abs(pred - actual)
    results.append([sid] + list(actual) + list(pred) + list(mae_err))

columns = ["sample_id", "Peak_Act", "Sow_Act", "Harv_Act", "Peak_Pred", "Sow_Pred", "Harv_Pred", "Peak_Err", "Sow_Err", "Harv_Err"]
df_res = pd.DataFrame(results, columns=columns)
df_res.to_csv("final_predictions.csv", index=False)
print("✅ Saved 'final_predictions.csv'")

# Show 5 random examples
print("\n--- Random Examples ---")
for idx in random.sample(range(len(df_res)), min(5, len(df_res))):
    row = df_res.iloc[idx]
    print(f"Sample: {row['sample_id']}")
    print(f"   Peak    | Act: {row['Peak_Act']:.1f} | Pred: {row['Peak_Pred']:.1f} | Err: {row['Peak_Err']:.1f}")
    print(f"   Sowing  | Act: {row['Sow_Act']:.1f} | Pred: {row['Sow_Pred']:.1f} | Err: {row['Sow_Err']:.1f}")
    print(f"   Harvest | Act: {row['Harv_Act']:.1f} | Pred: {row['Harv_Pred']:.1f} | Err: {row['Harv_Err']:.1f}")
    print("-" * 30)
