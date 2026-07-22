"""
Cross-validation 5-fold stratifiee (robustesse des resultats).
Produit AUC moyenne +/- ecart-type sur 5 decoupages differents.
Usage : python scripts/cross_validation.py --model resnet50 --epochs 12
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.build import build_model
from src.utils.seed import set_seed

class DataFrameDataset(Dataset):
    """Comme FractureDataset mais lit un DataFrame au lieu d'un CSV."""
    def __init__(self, df, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.cvtColor(cv2.imread(row["image_path"]), cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)["image"]
        return image, torch.tensor(row["label"], dtype=torch.float32)

def entrainer_un_fold(train_df, val_df, args, device):
    train_loader = DataLoader(DataFrameDataset(train_df, get_train_transforms()),
                              batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(DataFrameDataset(val_df, get_val_transforms()),
                            batch_size=args.batch, num_workers=2, pin_memory=True)

    labels = train_df["label"].values
    pos_weight = torch.tensor([(labels == 0).sum() / (labels == 1).sum()], device=device)

    model = build_model(args.model).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    meilleure_auc = 0.0
    for epoch in range(args.epochs):
        model.train()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        model.eval()
        probs, trues = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                with torch.cuda.amp.autocast():
                    out = model(images.to(device))
                probs.extend(torch.sigmoid(out).cpu().numpy().ravel())
                trues.extend(targets.numpy().ravel())
        auc = roc_auc_score(trues, probs)
        meilleure_auc = max(meilleure_auc, auc)
        print(f"    epoch {epoch+1}/{args.epochs} - AUC {auc:.4f}")
    return meilleure_auc

def run(args):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = pd.read_csv(ROOT / "data" / "processed" / "index.csv")
    print(f"Cross-validation {args.folds}-fold sur {len(index)} images\n")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    aucs = []
    for i, (idx_tr, idx_va) in enumerate(skf.split(index, index["label"]), start=1):
        print(f"--- Fold {i}/{args.folds} ---")
        auc = entrainer_un_fold(index.iloc[idx_tr], index.iloc[idx_va], args, device)
        aucs.append(auc)
        print(f"  => Fold {i} : AUC = {auc:.4f}\n")

    aucs = np.array(aucs)
    print("=" * 45)
    print(f"AUC par fold : {np.round(aucs, 4).tolist()}")
    print(f"AUC moyenne  : {aucs.mean():.4f} +/- {aucs.std():.4f}")
    print("=" * 45)

    sortie = ROOT / "outputs" / "tables" / f"cv_{args.model}.txt"
    sortie.parent.mkdir(parents=True, exist_ok=True)
    sortie.write_text(
        f"Cross-validation {args.folds}-fold - {args.model}\n"
        f"AUC par fold : {np.round(aucs, 4).tolist()}\n"
        f"AUC moyenne  : {aucs.mean():.4f} +/- {aucs.std():.4f}\n"
    )
    print(f"Resultats sauvegardes : {sortie}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet50")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--folds", type=int, default=5)
    run(p.parse_args())
