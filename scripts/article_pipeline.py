"""
Script maître pour l'article FracAtlas — tout pour UN modèle :
  entraînement final + évaluation + calibration (ECE, temperature scaling) +
  matrice de confusion + reliability diagram + cross-validation 5-fold.
Chaque résultat est sauvegardé dès qu'il est calculé.
Usage : python scripts/article_pipeline.py --model resnet50
"""
import argparse, json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.data.dataset import FractureDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.build import build_model
from src.utils.seed import set_seed

TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# ---------- Dataset depuis DataFrame (pour la CV) ----------
class DFDataset(Dataset):
    def __init__(self, df, tf):
        self.df = df.reset_index(drop=True); self.tf = tf
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.cvtColor(cv2.imread(r["image_path"]), cv2.COLOR_BGR2RGB)
        return self.tf(image=img)["image"], torch.tensor(r["label"], dtype=torch.float32)

def train_model(train_loader, val_loader, pos_weight, model_name, epochs, device):
    model = build_model(model_name).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()
    best_auc, best_state, no_imp = 0.0, None, 0
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        model.eval(); p, t = [], []
        with torch.no_grad():
            for x, y in val_loader:
                with torch.cuda.amp.autocast():
                    o = model(x.to(device))
                p.extend(torch.sigmoid(o).cpu().numpy().ravel()); t.extend(y.numpy().ravel())
        p = np.nan_to_num(np.array(p), nan=0.5)  # remplace les NaN eventuels
        try:
            auc = roc_auc_score(t, p)
        except ValueError:
            auc = 0.5
        if auc > best_auc:
            best_auc, best_state, no_imp = auc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= 5: break
    model.load_state_dict(best_state)
    return model, best_auc

def predict(model, loader, device):
    model.eval(); logits, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            o = model(x.to(device))
            logits.extend(o.cpu().numpy().ravel()); trues.extend(y.numpy().ravel())
    return np.array(logits), np.array(trues)

def ece_score(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1); ece = 0.0; pts = []
    for i in range(n_bins):
        m = (p > bins[i]) & (p <= bins[i+1])
        if m.sum() > 0:
            conf, acc = p[m].mean(), y[m].mean()
            ece += (m.sum()/len(p)) * abs(conf - acc); pts.append((conf, acc))
    return ece, pts

def temperature_scale(logits, y):
    """Trouve T qui minimise la log-vraisemblance (methode standard, Guo et al. 2017)."""
    eps = 1e-7
    best_T, best_nll = 1.0, 1e18
    for T in np.arange(0.5, 5.05, 0.05):
        p = 1/(1+np.exp(-logits/T))
        p = np.clip(p, eps, 1-eps)
        nll = -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
        if nll < best_nll: best_nll, best_T = nll, T
    return best_T

def run(args):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = args.model
    print(f"===== {m} sur {device} =====")
    results = {}

    splits = ROOT / "data" / "splits"
    tr_df = pd.read_csv(splits / "train.csv")
    pw = torch.tensor([(tr_df.label==0).sum()/(tr_df.label==1).sum()], device=device)

    tr = DataLoader(FractureDataset(splits/"train.csv", get_train_transforms()),
                    batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    va = DataLoader(FractureDataset(splits/"val.csv", get_val_transforms()),
                    batch_size=args.batch, num_workers=2, pin_memory=True)
    te = DataLoader(FractureDataset(splits/"test.csv", get_val_transforms()),
                    batch_size=args.batch, num_workers=2, pin_memory=True)

    # 1) Entraînement final
    print("[1/4] Entrainement final...")
    model, val_auc = train_model(tr, va, pw, m, args.epochs, device)
    torch.save(model.state_dict(), ROOT/"checkpoints"/f"best_{m}.pt")

    # 2) Évaluation test + matrice de confusion
    print("[2/4] Evaluation test...")
    logit_te, y_te = predict(model, te, device)
    p_te = 1/(1+np.exp(-logit_te))
    logit_va, y_va = predict(model, va, device)
    p_va = 1/(1+np.exp(-logit_va))
    fpr, tpr, thr = roc_curve(y_va, p_va)
    youden = thr[np.argmax(tpr - fpr)]
    pred = (p_te >= youden).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    results["auc_test"] = float(roc_auc_score(y_te, p_te))
    results["sensibilite"] = float(tp/(tp+fn)); results["specificite"] = float(tn/(tn+fp))
    results["vpp"] = float(tp/(tp+fp)); results["vpn"] = float(tn/(tn+fn))
    results["f1"] = float(f1_score(y_te, pred))
    results["confusion"] = {"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}
    # figure matrice de confusion
    plt.figure(figsize=(4,4))
    cm = np.array([[tn,fp],[fn,tp]])
    plt.imshow(cm, cmap="Blues"); plt.colorbar()
    for (i,j),v in np.ndenumerate(cm):
        plt.text(j,i,str(v),ha="center",va="center",fontsize=14)
    plt.xticks([0,1],["Normal","Fracture"]); plt.yticks([0,1],["Normal","Fracture"])
    plt.xlabel("Predit"); plt.ylabel("Reel"); plt.title(f"Matrice - {m}")
    plt.tight_layout(); plt.savefig(FIG/f"confusion_{m}.png", dpi=120); plt.close()

    # 3) Calibration + temperature scaling
    print("[3/4] Calibration...")
    ece_avant, pts_avant = ece_score(y_te, p_te)
    T = temperature_scale(logit_va, y_va)   # T calibree sur validation
    p_te_cal = 1/(1+np.exp(-logit_te/T))
    ece_apres, pts_apres = ece_score(y_te, p_te_cal)
    results["ece_avant"] = float(ece_avant); results["ece_apres"] = float(ece_apres); results["temperature"] = float(T)
    # reliability diagram avant/apres
    plt.figure(figsize=(6,6))
    plt.plot([0,1],[0,1],"k--",label="Parfait")
    plt.plot([c for c,a in pts_avant],[a for c,a in pts_avant],"o-",label=f"Avant (ECE={ece_avant:.3f})")
    plt.plot([c for c,a in pts_apres],[a for c,a in pts_apres],"s-",label=f"Apres TS (ECE={ece_apres:.3f})")
    plt.xlabel("Confiance predite"); plt.ylabel("Proportion reelle")
    plt.title(f"Calibration - {m}"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG/f"calibration_{m}.png", dpi=120); plt.close()

    # 4) Cross-validation 5-fold
    print("[4/4] Cross-validation 5-fold (le plus long)...")
    index = pd.read_csv(ROOT/"data"/"processed"/"index.csv")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []
    for k,(itr,iva) in enumerate(skf.split(index, index["label"]),1):
        dtr = DataLoader(DFDataset(index.iloc[itr], get_train_transforms()),
                         batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
        dva = DataLoader(DFDataset(index.iloc[iva], get_val_transforms()),
                         batch_size=args.batch, num_workers=2, pin_memory=True)
        _, a = train_model(dtr, dva, pw, m, args.cv_epochs, device)
        cv_aucs.append(float(a)); print(f"    fold {k}: AUC={a:.4f}")
    results["cv_aucs"] = cv_aucs
    results["cv_mean"] = float(np.mean(cv_aucs)); results["cv_std"] = float(np.std(cv_aucs))

    # Sauvegarde JSON
    out = TAB/f"article_{m}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nRESULTATS ({m}) :")
    print(json.dumps(results, indent=2))
    print(f"\nSauvegarde : {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet50")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--cv_epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=32)
    run(p.parse_args())
