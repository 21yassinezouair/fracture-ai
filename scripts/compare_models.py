"""
Comparaison statistique des modeles par le test de DeLong.
Repond a : les differences d'AUC sont-elles significatives ?
Produit aussi la figure des courbes ROC superposees.
Usage : python scripts/compare_models.py
"""
from pathlib import Path
import sys
import numpy as np
import scipy.stats
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.data.dataset import FractureDataset
from src.data.transforms import get_val_transforms
from src.models.build import build_model

MODELS = ["convnext_tiny", "resnet50", "vit_base_patch16_224", "efficientnet_b4"]

# ---------- Implementation du test de DeLong ----------
def compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2

def fastDeLong(preds_sorted, m):
    n = preds_sorted.shape[1] - m; k = preds_sorted.shape[0]
    pos = preds_sorted[:, :m]; neg = preds_sorted[:, m:]
    tx = np.array([compute_midrank(pos[r]) for r in range(k)])
    ty = np.array([compute_midrank(neg[r]) for r in range(k)])
    tz = np.array([compute_midrank(preds_sorted[r]) for r in range(k)])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def delong_test(y_true, prob_a, prob_b):
    order = (-y_true).argsort(); m = int(y_true.sum())
    preds = np.vstack((prob_a, prob_b))[:, order]
    aucs, cov = fastDeLong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return aucs, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * (1 - scipy.stats.norm.cdf(abs(z)))
    return aucs, p

# ---------- Recuperer les predictions de chaque modele ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
test_loader = DataLoader(FractureDataset(ROOT / "data/splits/test.csv", get_val_transforms()),
                         batch_size=32, num_workers=2)

def get_preds(name):
    model = build_model(name, pretrained=False).to(device)
    model.load_state_dict(torch.load(ROOT / "checkpoints" / f"best_{name}.pt", map_location=device))
    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            out = model(images.to(device))
            probs.extend(torch.sigmoid(out).cpu().numpy().ravel())
            trues.extend(targets.numpy().ravel())
    return np.array(trues), np.array(probs)

preds = {}
y_true = None
for name in MODELS:
    y_true, p = get_preds(name)
    preds[name] = p
    print(f"{name}: AUC = {roc_auc_score(y_true, p):.3f}")

# ---------- Comparaison du meilleur (ConvNeXt) vs les autres ----------
print("\n--- Test de DeLong (ConvNeXt vs autres) ---")
best = "convnext_tiny"
for name in MODELS:
    if name == best:
        continue
    _, p = delong_test(y_true.astype(int), preds[best], preds[name])
    signif = "SIGNIFICATIF (p<0.05)" if p < 0.05 else "non significatif"
    print(f"ConvNeXt vs {name:22s} : p = {p:.4f}  -> {signif}")

# ---------- Figure : ROC superposees ----------
plt.figure(figsize=(7, 7))
for name in MODELS:
    fpr, tpr, _ = roc_curve(y_true, preds[name])
    auc = roc_auc_score(y_true, preds[name])
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("1 - Specificite"); plt.ylabel("Sensibilite")
plt.title("Comparaison des architectures - courbes ROC (test)")
plt.legend(loc="lower right")
plt.savefig(ROOT / "outputs" / "figures" / "roc_comparison.png", dpi=120)
print("\nFigure comparative sauvegardee : outputs/figures/roc_comparison.png")
