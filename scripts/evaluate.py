"""
Evaluation rigoureuse sur le test set (standards STARD 2015 / CLAIM 2024).
Calcule AUC + IC95%, sensibilite, specificite, VPP, VPN, F1, matrice de confusion,
et sauvegarde la courbe ROC.
Usage : python scripts/evaluate.py --model resnet50
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             precision_score, recall_score, f1_score)

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.data.dataset import FractureDataset
from src.data.transforms import get_val_transforms
from src.models.build import build_model
from src.utils.seed import set_seed

def bootstrap_auc(y_true, y_prob, n=1000, seed=42):
    """Intervalle de confiance 95% de l'AUC par bootstrap."""
    rng = np.random.RandomState(seed)
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    aucs = []
    for _ in range(n):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

def predict(model, loader, device):
    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            out = model(images)
            probs.extend(torch.sigmoid(out).cpu().numpy().ravel())
            trues.extend(targets.numpy().ravel())
    return np.array(trues), np.array(probs)

def run(args):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = ROOT / "data" / "splits"
    out = ROOT / "outputs"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    # --- Seuil optimal (Youden) calcule sur la VALIDATION ---
    model = build_model(args.model, pretrained=False).to(device)
    model.load_state_dict(torch.load(ROOT / "checkpoints" / f"best_{args.model}.pt",
                                     map_location=device))
    val_loader = DataLoader(FractureDataset(splits / "val.csv", get_val_transforms()),
                            batch_size=32, num_workers=2)
    yv, pv = predict(model, val_loader, device)
    fpr, tpr, thr = roc_curve(yv, pv)
    youden = thr[np.argmax(tpr - fpr)]
    print(f"Seuil optimal (Youden, calcule sur validation) : {youden:.3f}")

    # --- Evaluation sur le TEST (une seule fois) ---
    test_loader = DataLoader(FractureDataset(splits / "test.csv", get_val_transforms()),
                             batch_size=32, num_workers=2)
    yt, pt = predict(model, test_loader, device)
    pred = (pt >= youden).astype(int)

    auc = roc_auc_score(yt, pt)
    lo, hi = bootstrap_auc(yt, pt)
    tn, fp, fn, tp = confusion_matrix(yt, pred).ravel()
    se = tp / (tp + fn)          # sensibilite
    sp = tn / (tn + fp)          # specificite
    vpp = tp / (tp + fp)         # valeur predictive positive
    vpn = tn / (tn + fn)         # valeur predictive negative
    f1 = f1_score(yt, pred)

    # --- Affichage + sauvegarde du tableau ---
    lignes = [
        f"AUC          : {auc:.3f}  (IC95% {lo:.3f}-{hi:.3f})",
        f"Sensibilite  : {se:.3f}",
        f"Specificite  : {sp:.3f}",
        f"VPP          : {vpp:.3f}",
        f"VPN          : {vpn:.3f}",
        f"F1-score     : {f1:.3f}",
        f"Matrice : TN={tn} FP={fp} FN={fn} TP={tp}",
    ]
    print("\n--- RESULTATS TEST (" + args.model + ") ---")
    for l in lignes:
        print(l)
    (out / "tables" / f"results_{args.model}.txt").write_text("\n".join(lignes))

    # --- Courbe ROC ---
    fpr, tpr, _ = roc_curve(yt, pt)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"{args.model} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("1 - Specificite"); plt.ylabel("Sensibilite")
    plt.title("Courbe ROC - test set"); plt.legend()
    plt.savefig(out / "figures" / f"roc_{args.model}.png", dpi=120)
    print(f"\nCourbe ROC sauvegardee.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet50")
    run(p.parse_args())
