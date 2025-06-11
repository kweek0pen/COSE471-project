# Complete pipeline: DataLoader, 8-fold CV, CLIP fusion model, training, Grad-CAM & embedding visualization

"""
Dependencies:
  - torch, torchvision, clip (OpenAI CLIP)
  - numpy, pandas, scikit-learn, matplotlib, pillow
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import clip
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1) Configuration
root_dir = Path('drive/MyDrive/Colab Notebooks/datas/Stimulus Images')   # 수정: 실제 데이터 경로
n_splits = 8
batch_size = 16
num_epochs = 5
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2) Build DataFrame of image paths, categories, labels
records = []
for label_dir in root_dir.iterdir():
    if label_dir.is_dir():
        label = 1 if label_dir.name.lower() == "vme" else 0
        for img_path in label_dir.glob("*.jpg"):
            # 예: 'Apple_Original.jpg' → 'Apple'
            category = img_path.stem.split('_')[0]
            records.append({
                'path': str(img_path),
                'category': category,
                'label': label
            })
df = pd.DataFrame(records)

# 3) Transforms
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4815,0.4578,0.4082),(0.2686,0.2613,0.2758))
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4815,0.4578,0.4082),(0.2686,0.2613,0.2758))
])

# 4) Dataset
class VMEDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(row.label, dtype=torch.float32)
        cat = row.category
        return img, label, cat

skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(
        skf.split(df, df['label']), start=1):

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    train_ds = VMEDataset(train_df, train_tf)
    val_ds   = VMEDataset(val_df,   val_tf)

    train_loader = DataLoader(train_ds, batch_size=16,
                              shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=16,
                              shuffle=False, num_workers=4)

    # 폴드별 데이터 분포 확인
    print(f"--- Fold {fold} ---")
    print("Train labels:", train_df.label.value_counts().to_dict())
    print("Val   labels:",   val_df.label.value_counts().to_dict())

# 5) Prompt maker
def make_prompt(cat):
    return f"A clear icon of {cat}."

# 6) Model definition
class CLIPFusionClassifier(nn.Module):
    def __init__(self, clip_model, hidden_dim=256):
        super().__init__()
        self.clip = clip_model
        # RN50’s visual encoder produces a 1024-dim embedding:
        embed_dim = clip_model.visual.output_dim  

        # now use that dynamically instead of “512”
        self.img_proj = nn.Linear(embed_dim, embed_dim)
        self.txt_proj = nn.Linear(embed_dim, embed_dim)
        self.fuse     = nn.Sequential(
            nn.Linear(embed_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, images, text_tokens):
        img_emb = self.clip.encode_image(images).float()
        txt_emb = self.clip.encode_text(text_tokens).float()

        # both encode_image and encode_text return (B, embed_dim)
        img_emb = self.img_proj(img_emb)
        txt_emb = self.txt_proj(txt_emb)
        fused   = torch.cat([img_emb, txt_emb], dim=-1)   # (B, 2*embed_dim)
        logits  = self.fuse(fused).squeeze(-1)             # (B,)
        return fused, logits

# 7) Load CLIP and instantiate model
clip_model, preprocess = clip.load("RN50", device=device)
model = CLIPFusionClassifier(clip_model).to(device)
# Freeze CLIP
for p in model.clip.parameters(): p.requires_grad = False
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

# 8) Cross-validation training
"""skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

    for f, (train_idx, val_idx) in enumerate(
        skf.split(df, df['label']), start=1):

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    train_ds = VMEDataset(train_df, train_tf)
    val_ds   = VMEDataset(val_df,   val_tf)

    train_loader = DataLoader(train_ds, batch_size=16,
                              shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=16,
                              shuffle=False, num_workers=4)

    # 폴드별 데이터 분포 확인
    print(f"--- Fold {f} ---")
    print("Train labels:", train_df.label.value_counts().to_dict())
    print("Val   labels:",   val_df.label.value_counts().to_dict())"""

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_metrics = []
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label']), start=1):
    print(f"=== Fold {fold}/{n_splits} ===")
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    train_ds = VMEDataset(train_df, train_tf)
    val_ds   = VMEDataset(val_df, val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Train epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels, cats in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            tokens = clip.tokenize([make_prompt(c) for c in cats]).to(device)

            _, logits = model(imgs, tokens)
            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)

            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step()
        epoch_loss = running_loss / len(train_ds)
        print(f" Epoch {epoch+1}/{num_epochs} – Train Loss: {epoch_loss:.4f}")

    # Validation
    model.eval()
    ys, ps, preds = [], [], []
    with torch.no_grad():
        for imgs, labels, cats in val_loader:
            imgs=imgs.to(device)
            labels=labels.to(device)
            tokens=clip.tokenize([make_prompt(c) for c in cats]).to(device)

            _, logits=model(imgs, tokens)
            probs = torch.sigmoid(logits)

            ys.extend(labels.cpu().numpy())
            ps.extend(probs.cpu().numpy())
            preds.extend((probs.cpu().numpy() >= 0.5).astype(int))

    auc = roc_auc_score(ys, ps)
    preds = (np.array(ps)>=0.5).astype(int)
    f1 = f1_score(ys, preds)
    acc  = accuracy_score(ys, preds)
    print(f"Fold {fold+1} AUC: {auc:.3f}, F1: {f1:.3f}, Accuracy: {acc:.3f}")
    fold_metrics.append((auc, f1))


"""
# 9) Grad-CAM visualization for a sample
# Pick first val sample
sample_img, _, sample_cat = val_ds[0]
orig = Image.open(val_df.iloc[0].path).convert('RGB').resize((224,224))
orig_np = np.array(orig)/255.0
img_tensor = sample_img.unsqueeze(0).to(device)
text_token = clip.tokenize([make_prompt(sample_cat)]).to(device)

# Hook for last conv of RN50
# Setup hooks
gradients = []
activations = []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

# Hook to the last residual block
target_layer = model.clip.visual.layer4[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# Get one sample from val_ds
sample_img, _, sample_cat = val_ds[0]
img_tensor = sample_img.unsqueeze(0).to(device)
text_token = clip.tokenize([make_prompt(sample_cat)]).to(device)

# Forward + backward pass
model.zero_grad()
_, logits = model(img_tensor, text_token)
logit = logits[0]
logit.backward()

# Now hooks should have fired
if not gradients or not activations:
    raise RuntimeError("Grad-CAM hooks did not capture any gradients/activations.")

grad = gradients[0][0].cpu().numpy()
act  = activations[0][0].cpu().numpy()
weights = np.mean(grad, axis=(1, 2))
cam = np.sum(weights[:, None, None] * act, axis=0)
cam = np.maximum(cam, 0)
cam /= cam.max()


# Plot Grad-CAM
plt.figure(figsize=(5,5))
plt.imshow(orig_np)
plt.imshow(cam, cmap='jet', alpha=0.5)
plt.title(f"Grad-CAM: {sample_cat}")
plt.axis('off')"""

# 10) Embedding extraction and PCA/t-SNE
all_emb, all_lbl = [], []
full_ds = VMEDataset(df, val_tf)
full_loader = DataLoader(full_ds, batch_size=32, shuffle=False)
with torch.no_grad():
    for imgs, labels, cats in full_loader:
        imgs = imgs.to(device)
        tokens = clip.tokenize([make_prompt(c) for c in cats]).to(device)
        fused, _ = model(imgs, tokens)
        all_emb.append(fused.cpu().numpy()); all_lbl.extend(labels.numpy())
emb = np.concatenate(all_emb)
# PCA
pca = PCA(n_components=2).fit_transform(emb)
# t-SNE
tsne = TSNE(n_components=2, perplexity=30).fit_transform(emb)

# Plot embeddings
plt.figure(figsize=(6,5))
for lbl,marker,col in [(0,'o','blue'),(1,'^','red')]:
    idx = np.array(all_lbl)==lbl
    plt.scatter(pca[idx,0], pca[idx,1], marker=marker, c=col, label=f"VME={lbl}", alpha=0.6)
plt.title("PCA Embedding")
plt.legend()
plt.show()

plt.figure(figsize=(6,5))
for lbl,marker,col in [(0,'o','blue'),(1,'^','red')]:
    idx = np.array(all_lbl)==lbl
    plt.scatter(tsne[idx,0], tsne[idx,1], marker=marker, c=col, label=f"VME={lbl}", alpha=0.6)
plt.title("t-SNE Embedding")
plt.legend()
plt.show()