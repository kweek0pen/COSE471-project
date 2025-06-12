import os
import cv2
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


root_dir = Path('drive/MyDrive/Colab Notebooks/datas/Stimulus Images')
n_splits = 8
batch_size = 16
num_epochs = 5
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"


records = []
for label_dir in root_dir.iterdir():
    if label_dir.is_dir():
        label = 1 if label_dir.name.lower() == "vme" else 0
        for img_path in label_dir.glob("*.jpg"):
            category = img_path.stem.split('_')[0]
            records.append({
                'path': str(img_path),
                'category': category,
                'label': label
            })
df = pd.DataFrame(records)

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


def make_prompt(cat):
    return f"A clear icon of {cat}."

class CLIPFusionClassifier(nn.Module):
    def __init__(self, clip_model, hidden_dim=256):
        super().__init__()
        self.clip = clip_model
        embed_dim = clip_model.visual.output_dim

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

        img_emb = self.img_proj(img_emb)
        txt_emb = self.txt_proj(txt_emb)
        fused   = torch.cat([img_emb, txt_emb], dim=-1)
        logits  = self.fuse(fused).squeeze(-1)
        return fused, logits

clip_model, preprocess = clip.load("RN50", device=device)
model = CLIPFusionClassifier(clip_model).to(device)

for p in model.clip.parameters(): p.requires_grad = False
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

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

weight_path = "vme_classifier_weights_fold{fold}.pth"
torch.save(model.state_dict(), weight_path)

print(f"Model weights saved to: {weight_path}")

model.eval()

target_layer = model.clip.visual.layer4[-1]
for p in target_layer.parameters():
    p.requires_grad = True

gradients = []
activations = []

def forward_hook(module, inp, out):
    activations.append(out)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

preproc = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4815,0.4578,0.4082),
                         (0.2686,0.2613,0.2758))
])

os.makedirs("gradcam_outputs1", exist_ok=True)

for idx, row in df.iterrows():
    gradients.clear()
    activations.clear()

    img = Image.open(row.path).convert("RGB")
    inp = preproc(img).unsqueeze(0).to(device)
    inp.requires_grad_()

    txt = clip.tokenize([make_prompt(row.category)]).to(device)

    model.zero_grad()
    _, logits = model(inp, txt)
    score = logits.squeeze()

    score.backward()

    if not gradients or not activations:
        raise RuntimeError("No grads/acts—check that layer wasn’t frozen.")

    grad = gradients[0][0].detach().cpu().numpy()
    act  = activations[0][0].detach().cpu().numpy()
    weights = grad.mean(axis=(1,2))
    cam = np.maximum((weights[:,None,None]*act).sum(axis=0), 0)
    cam /= cam.max()

    orig = Image.open(row.path).convert("RGB")
    orig_w, orig_h = orig.size
    orig_np = np.array(orig)

    cam_resized = cv2.resize(
        (cam * 255).astype(np.uint8),
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR
    )

    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)

    blended = cv2.addWeighted(
        orig_np[..., ::-1], 0.6,
        heatmap,         0.4,
        gamma=0
    )

    out = Image.fromarray(blended[..., ::-1])
    out.save(f"gradcam_outputs/{row.category}_{idx}_overlay_big.png")


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

pca = PCA(n_components=2).fit_transform(emb)

tsne = TSNE(n_components=2, perplexity=30).fit_transform(emb)

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