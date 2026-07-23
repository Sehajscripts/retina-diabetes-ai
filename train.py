import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np

# ==========================================
# 1. MULTI-CLASS DATASET LOADER ENGINE
# ==========================================
class MultiClassRetinalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

        # Input image transformation (ImageNet parameters)
        self.img_transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Load raw scan 
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.img_transform(image)
        
        # Load mask and map to 256x256 size layout
        mask = Image.open(mask_path).convert("RGB")
        mask_resized = mask.resize(self.img_size, Image.NEAREST)
        mask_np = np.array(mask_resized)

        # Separate the 3 composite channels from the RGB image array
        # This converts a single image file into 3 distinct binary arrays
        hem_layer = (mask_np[:, :, 0] > 127).astype(np.float32)
        exu_layer = (mask_np[:, :, 1] > 127).astype(np.float32)
        ane_layer = (mask_np[:, :, 2] > 127).astype(np.float32)

        # Stack layers on top of each other to produce a 3-channel tracking tensor
        mask_stack = np.stack([hem_layer, exu_layer, ane_layer], axis=0)
        mask_tensor = torch.from_numpy(mask_stack)

        return image_tensor, mask_tensor

# ==========================================
# 2. RUN TIME CONFIGURATION SETUP
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 5

TRAIN_IMG_DIR = "C:/Project505/dataset/images"
TRAIN_MASK_DIR = "C:/Project505/dataset/masks"

# ==========================================
# 3. INITIALIZE TRAINING FRAMEWORK
# ==========================================
print("🔄 Indexing multi-class asset data directories...")
dataset = MultiClassRetinalDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

print(f"🧠 Assembling U-Net Neural Layer Stack (Target: {DEVICE.upper()})...")
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3                        # CONFIGURATION FOR 3 MULTI-CLASS OUTPUT PATHS
).to(DEVICE)

# Binary Cross Entropy with Logits performs cross-channel loss calculation automatically
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 4. TRAINING LOOP ROUTINE
# ==========================================
print(f"🚀 Math optimization engine activated across {EPOCHS} Epochs...")

best_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    epoch_loss = running_loss / len(train_loader)
    print(f"📊 Epoch [{epoch}/{EPOCHS}] ---> Combined Multi-Class Loss: {epoch_loss:.4f}")
    
    # Save optimized model state coordinates to disk
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "best_model_multiclass.pth")
        print("💾 Optimal weights checkpoint exported as 'best_model_multiclass.pth'!")

print(f"\n✅ Done! Multi-Class brain file exported safely to: {os.path.abspath('best_model_multiclass.pth')}")
