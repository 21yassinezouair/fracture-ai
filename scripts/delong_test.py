"""
Entraine les 4 modeles, sauvegarde leurs predictions sur le test,
puis calcule le test de DeLong (comparaison statistique des AUC).
Les predictions sont sauvegardees -> on ne les reperd pas.
Usage : python scripts/delong_test.py
"""
from pathlib import Path
import sys, json
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.data.dataset import FractureDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.build import build_model
from src.utils.seed import set_seed

MODELS = ["convnext_tiny", "resnet50", "vit_base_patch16_224", "efficientnet_b4"]
TAB = ROOT / "outputs" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

# ---------- DeLong ----------
def compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5*(i+j-1)+1; i = j
    T2 = np.empty(N); T2[J] = T; return T2

def fastDeLong(preds_sorted, m):
    n = preds_sorted.shape[1]-m; k = preds_sorted.shape[0]
    pos = preds_sorted[:, :m]; neg = preds_sorted[:, m:]
    tx = np.array([compute_midrank(pos[r]) for r in range(k)])
    ty = np.array([compute_midrank(neg[r]) for r in range(k)])
    tz = np.array([compute_midrank(preds_sorted[r]) for r in range(k)])
    aucs = (tz[:, :m].sum(axis=1)/m - (m+1)/2)/n
    v01 = (tz[:, :m]-tx)/n; v10 = 1-(tz[:, m:]-ty)/m
    sx = np.cov(v01); sy = np.cov(v10)
    return aucs, sx/m + sy/n

def delong_p(y, pa, pb):
    order = (-y).argsort(); m = int(y.sum())
    preds = np.vstack((pa, pb))[:, order]
    aucs, cov = fastDeLong(preds, m)
    var = cov[0,0]+cov[1,1]-2*cov[0,1]
    if var <= 0: return 1.0
    z = (aucs[0]-aucs[1])/np.sqrt(var)
    return 2*(1-scipy.stats.norm.cdf(abs(z)))

# ---------- Entrainement rapide + prediction ----------
def train_and_predict(name, device):
    set_seed(42)
    tr_df_path = ROOT/"data/splits/train.csv"
    import pandas as pd
    tr_df = pd.read_csv(tr_df_path)
    pw = torch.tensor([(tr_df.label==0).sum()/(tr_df.label==1).sum()], device=device)
    tr = DataLoader(FractureDataset(tr_df_path, get_train_transforms()),
                    batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    te = DataLoader(FractureDataset(ROOT/"data/splits/test.csv", get_val_transforms()),
                    batch_size=32, num_workers=2)
    model = build_model(name).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    for ep in range(12):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update()
        print(f"    {name} epoch {ep+1}/12")
    model.eval(); probs, trues = [], []
    with torch.no_grad():
        for x, y in te:
            o = model(x.to(device))
            probs.extend(torch.sigmoid(o).cpu().numpy().ravel()); trues.extend(y.numpy().ravel())
    return np.array(probs), np.array(trues)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds = {}; y_true = None
    for name in MODELS:
        print(f"=== {name} ===")
        p, y = train_and_predict(name, device)
        preds[name] = p.tolist(); y_true = y
        print(f"    AUC = {roc_auc_score(y, p):.3f}")
    # Sauvegarder les predictions (pour ne plus les reperdre)
    (TAB/"predictions.json").write_text(json.dumps({"y_true": y_true.tolist(), "preds": preds}))
    print("\nPredictions sauvegardees dans outputs/tables/predictions.json")

    # DeLong : ConvNeXt vs les autres
    y = np.array(y_true).astype(int)
    best = "convnext_tiny"
    print("\n--- Test de DeLong (ConvNeXt vs autres) ---")
    lignes = []
    for name in MODELS:
        if name == best: continue
        p = delong_p(y, np.array(preds[best]), np.array(preds[name]))
        signif = "SIGNIFICATIF" if p < 0.05 else "non significatif"
        print(f"ConvNeXt vs {name:24s} : p = {p:.4f}  -> {signif}")
        lignes.append(f"ConvNeXt vs {name}: p={p:.4f} ({signif})")
    (TAB/"delong.txt").write_text("\n".join(lignes))
    print("\nResultats sauvegardes dans outputs/tables/delong.txt")

if __name__ == "__main__":
    main()
