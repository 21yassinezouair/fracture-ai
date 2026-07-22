"""
Interpretabilite par Grad-CAM (objectif OS6).
Genere une grille : 4 vrais positifs, 4 faux positifs, 4 faux negatifs.
Usage : python scripts/gradcam.py --model resnet50
"""
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.data.dataset import FractureDataset
from src.data.transforms import get_val_transforms
from src.models.build import build_model
from src.utils.seed import set_seed

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
SEUIL = 0.5

def run(args):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    figs = ROOT / "outputs" / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    # --- Charger le modele entraine ---
    model = build_model(args.model, pretrained=False).to(device)
    model.load_state_dict(torch.load(ROOT / "checkpoints" / f"best_{args.model}.pt",
                                     map_location=device))
    model.eval()

    # --- Predire sur tout le test set ---
    ds = FractureDataset(ROOT / "data/splits/test.csv", get_val_transforms())
    loader = DataLoader(ds, batch_size=32, num_workers=2)
    probs, trues = [], []
    with torch.no_grad():
        for images, targets in loader:
            out = model(images.to(device))
            probs.extend(torch.sigmoid(out).cpu().numpy().ravel())
            trues.extend(targets.numpy().ravel())
    probs, trues = np.array(probs), np.array(trues)
    preds = (probs >= SEUIL).astype(int)

    # --- Selectionner 4 cas de chaque categorie ---
    tp = np.where((trues == 1) & (preds == 1))[0][:4]   # vrais positifs
    fp = np.where((trues == 0) & (preds == 1))[0][:4]   # faux positifs
    fn = np.where((trues == 1) & (preds == 0))[0][:4]   # faux negatifs
    print(f"Cas trouves -> TP:{len(tp)}  FP:{len(fp)}  FN:{len(fn)}")

    # --- Grad-CAM sur la derniere couche convolutionnelle ---
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

    def heatmap(idx):
        image, _ = ds[idx]
        tensor = image.unsqueeze(0).to(device)
        grayscale = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(0)])[0]
        rgb = image.permute(1, 2, 0).numpy()
        rgb = (rgb * STD + MEAN).clip(0, 1).astype(np.float32)
        return show_cam_on_image(rgb, grayscale, use_rgb=True), probs[idx]

    # --- Construire la figure ---
    groupes = [("Vrai Positif", tp), ("Faux Positif", fp), ("Faux Negatif", fn)]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for ligne, (titre, indices) in enumerate(groupes):
        for col in range(4):
            ax = axes[ligne, col]
            if col < len(indices):
                img, p = heatmap(indices[col])
                ax.imshow(img)
                ax.set_title(f"{titre}\nproba = {p:.2f}", fontsize=10)
            ax.axis("off")
    plt.suptitle(f"Grad-CAM - {args.model} (zones chaudes = regions determinantes)",
                 fontsize=14)
    plt.tight_layout()
    sortie = figs / f"gradcam_{args.model}.png"
    plt.savefig(sortie, dpi=110)
    print(f"Figure sauvegardee : {sortie}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet50")
    run(p.parse_args())
