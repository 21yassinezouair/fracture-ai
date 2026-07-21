"""
Entrainement d'un modele de detection de fractures.
Usage : python scripts/train.py --model resnet50 --epochs 15
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import wandb

# Racine du projet = dossier parent de scripts/ (marche partout)
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.data.dataset import FractureDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.build import build_model
from src.utils.seed import set_seed

def run(args):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    splits = ROOT / "data" / "splits"
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = FractureDataset(splits / "train.csv", get_train_transforms())
    val_ds = FractureDataset(splits / "val.csv", get_val_transforms())
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    labels = train_ds.df["label"].values
    pos_weight = torch.tensor([(labels == 0).sum() / (labels == 1).sum()], device=device)
    print(f"pos_weight : {pos_weight.item():.2f}")

    model = build_model(args.model).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    wandb.init(project="fracture-ai", name=args.model, config=vars(args))

    best_auc = 0.0
    patience, no_improve = 5, 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().ravel()
                preds.extend(probs)
                trues.extend(targets.numpy().ravel())
        auc = roc_auc_score(trues, preds)
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} | loss {avg_loss:.4f} | val AUC {auc:.4f}")
        wandb.log({"epoch": epoch+1, "train_loss": avg_loss, "val_auc": auc})

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), ckpt_dir / f"best_{args.model}.pt")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    print(f"\nMeilleure AUC validation : {best_auc:.4f}")
    wandb.finish()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet50")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=32)
    run(p.parse_args())
