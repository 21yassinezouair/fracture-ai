"""
Analyse des performances par localisation anatomique (hypothese H2).
FracAtlas fournit : hand, leg, hip, shoulder.
Usage : python scripts/analyse_localisation.py --model resnet50
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.data.dataset import FractureDataset
from src.data.transforms import get_val_transforms
from src.models.build import build_model
from src.utils.seed import set_seed

def run(args):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Predictions sur le test set ---
    test_csv = ROOT / "data/splits/test.csv"
    ds = FractureDataset(test_csv, get_val_transforms())
    loader = DataLoader(ds, batch_size=32, num_workers=2)

    model = build_model(args.model, pretrained=False).to(device)
    model.load_state_dict(torch.load(ROOT / "checkpoints" / f"best_{args.model}.pt",
                                     map_location=device))
    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for images, targets in loader:
            out = model(images.to(device))
            probs.extend(torch.sigmoid(out).cpu().numpy().ravel())
            trues.extend(targets.numpy().ravel())

    test = pd.read_csv(test_csv)
    test["prob"] = probs
    test["true"] = trues

    # --- Recuperer la localisation depuis le CSV original de FracAtlas ---
    src = pd.read_csv(args.dataset_csv)
    src["image_id_court"] = src["image_id"]
    test["image_id_court"] = test["image_path"].apply(lambda p: Path(p).name)
    fusion = test.merge(src[["image_id_court", "hand", "leg", "hip", "shoulder"]],
                        on="image_id_court", how="left")

    # --- AUC par localisation ---
    lignes = []
    print(f"\n--- Performances par localisation ({args.model}) ---")
    for loc in ["hand", "leg", "hip", "shoulder"]:
        sous = fusion[fusion[loc] == 1]
        n = len(sous)
        n_frac = int(sous["true"].sum())
        if n_frac < 5 or (n - n_frac) < 5:
            ligne = f"{loc:10s} : n={n:4d}  fractures={n_frac:3d}  -> effectif insuffisant"
        else:
            auc = roc_auc_score(sous["true"], sous["prob"])
            ligne = f"{loc:10s} : n={n:4d}  fractures={n_frac:3d}  AUC = {auc:.3f}"
            lignes.append((loc, n, n_frac, auc))
        print(ligne)

    # --- Sauvegarder le tableau ---
    sortie = ROOT / "outputs" / "tables" / f"localisation_{args.model}.txt"
    sortie.parent.mkdir(parents=True, exist_ok=True)
    texte = f"Performances par localisation - {args.model}\n"
    for loc, n, nf, auc in lignes:
        texte += f"{loc}: n={n}, fractures={nf}, AUC={auc:.3f}\n"
    sortie.write_text(texte)
    print(f"\nTableau sauvegarde : {sortie}")

    # --- Figure ---
    if lignes:
        noms = [l[0] for l in lignes]
        aucs = [l[3] for l in lignes]
        plt.figure(figsize=(7, 5))
        barres = plt.bar(noms, aucs, color="steelblue")
        plt.axhline(0.90, color="red", linestyle="--", label="Seuil H0 = 0.90")
        plt.ylim(0.5, 1.0)
        plt.ylabel("AUC")
        plt.title(f"AUC par localisation anatomique - {args.model}")
        for barre, auc in zip(barres, aucs):
            plt.text(barre.get_x() + barre.get_width()/2, auc + 0.01,
                     f"{auc:.3f}", ha="center")
        plt.legend()
        plt.tight_layout()
        fig = ROOT / "outputs" / "figures" / f"localisation_{args.model}.png"
        plt.savefig(fig, dpi=120)
        print(f"Figure sauvegardee : {fig}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet50")
    p.add_argument("--dataset_csv",
                   default="/kaggle/input/datasets/tommyngx/fracatlas/FracAtlas/dataset.csv")
    run(p.parse_args())
