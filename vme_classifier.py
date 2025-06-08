import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ----------------------------
# 1. Configuration & Helpers
# ----------------------------
DATA_DIR = "StimulusImages"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def sup_contrastive_loss(features, labels, temperature=0.07):
    """
    Supervised contrastive loss (Khosla et al. 2020).
    features: Tensor of shape (2N, D), normalized
    labels:   Tensor of shape (2N,)
    """
    device = features.device
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    logits = torch.matmul(features, features.T) / temperature
    logits_mask = torch.ones_like(mask) - torch.eye(labels.size(0), device=device)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
    loss = -mean_log_prob_pos.mean()
    return loss

# ----------------------------
# 2. Data & Augmentations
# ----------------------------
contrastive_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

classifier_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

class ContrastiveDataset(Dataset):
    def __init__(self, root_dir):
        self.dataset = datasets.ImageFolder(root_dir, transform=contrastive_transforms)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img2, _ = self.dataset[idx]  # second augmented view
        return img, img2, label

class SimpleDataset(Dataset):
    def __init__(self, root_dir):
        self.dataset = datasets.ImageFolder(root_dir, transform=classifier_transforms)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx]

# ----------------------------
# 3. Model Definitions
# ----------------------------
class FrozenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # drop final FC layer
        self.encoder = nn.Sequential(*modules)
        for p in self.encoder.parameters():
            p.requires_grad = False
    def forward(self, x):
        x = self.encoder(x)        # (N, 2048, 1, 1)
        return x.view(x.size(0), -1)  # (N, 2048)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

class ClassifierHead(nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

# ----------------------------
# 4. Training Functions
# ----------------------------
def train_contrastive(encoder, projector, loader, optimizer, epochs=10):
    encoder.train()
    projector.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x1, x2, labels in tqdm(loader, desc=f"Contrastive Epoch {epoch+1}/{epochs}"):
            x1, x2, labels = x1.to(DEVICE), x2.to(DEVICE), labels.to(DEVICE)
            h1, h2 = encoder(x1), encoder(x2)
            z1, z2 = projector(h1), projector(h2)
            feats = torch.cat([z1, z2], dim=0)
            labs = torch.cat([labels, labels], dim=0)
            loss = sup_contrastive_loss(feats, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss {total_loss/len(loader):.4f}")

def train_classifier(encoder, classifier, loader, optimizer, epochs=5):
    encoder.eval()
    classifier.train()
    criterion = nn.BCELoss()
    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, labels in tqdm(loader, desc=f"CLS Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)
            with torch.no_grad():
                feats = encoder(imgs)
            preds = classifier(feats)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss {total_loss/len(loader):.4f}")

# ----------------------------
# 5. Main Pipeline
# ----------------------------
def pca_torch(X, n_components):
    """
    PCA via SVD on torch.Tensor.
    Returns: (X_reduced, components)
    """
    X_centered = X - X.mean(dim=0)
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
    comps = Vt[:n_components]
    X_red = X_centered @ comps.T
    return X_red, comps

def main():
    # 5.1 Contrastive fine-tuning
    ds_con = ContrastiveDataset(DATA_DIR)
    loader_con = DataLoader(ds_con, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    encoder = FrozenEncoder().to(DEVICE)
    projector = ProjectionHead().to(DEVICE)
    opt_con = torch.optim.AdamW(
        list(projector.parameters()) + list(encoder.encoder[-1].parameters()),
        lr=1e-5
    )
    train_contrastive(encoder, projector, loader_con, opt_con, epochs=20)

    # 5.2 Train MLP head
    ds_clf = SimpleDataset(DATA_DIR)
    loader_clf = DataLoader(ds_clf, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    classifier = ClassifierHead().to(DEVICE)
    opt_clf = torch.optim.AdamW(classifier.parameters(), lr=1e-4)
    train_classifier(encoder, classifier, loader_clf, opt_clf, epochs=10)

    # 5.3 Extract embeddings & logistic regression on PCA features
    encoder.eval()
    feats_list, labels_list = [], []
    for img, label in loader_clf:
        with torch.no_grad():
            z = encoder(img.to(DEVICE)).cpu()
        feats_list.append(z)
        labels_list.append(label)
    X = torch.vstack(feats_list)
    y = torch.tensor(torch.cat(labels_list), dtype=torch.float32).unsqueeze(1)

    # PCA to 50 dims
    X_pca, _ = pca_torch(X, n_components=50)

    # Train torch-based logistic regression
    class LogRegTorch(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.linear = nn.Linear(in_dim, 1)
        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    logreg = LogRegTorch(in_dim=50).to(DEVICE)
    opt_lr = torch.optim.AdamW(logreg.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    for epoch in range(100):
        logreg.train()
        opt_lr.zero_grad()
        preds = logreg(X_pca.to(DEVICE))
        loss = criterion(preds, y.to(DEVICE))
        loss.backward()
        opt_lr.step()
        if epoch % 10 == 0:
            print(f"LogReg Epoch {epoch}: loss {loss.item():.4f}")

    # Final accuracy
    logreg.eval()
    with torch.no_grad():
        probs = logreg(X_pca.to(DEVICE)).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    acc = (preds.flatten() == y.numpy().flatten()).mean()
    print(f"Logistic Regression accuracy on PCA features: {acc:.3f}")

if __name__ == "__main__":
    main()