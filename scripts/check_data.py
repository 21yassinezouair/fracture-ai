"""
Sanity check : affiche 4 images normales + 4 fractures cote a cote.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path.home() / "fracture-ai"))
from src.data.dataset import FractureDataset
from src.data.transforms import get_train_transforms

SPLITS = Path.home() / "fracture-ai" / "data" / "splits"
OUT = Path.home() / "fracture-ai" / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

df = pd.read_csv(SPLITS / "train.csv")
# 4 fractures + 4 normales
idx_frac = df[df.label == 1].index[:4].tolist()
idx_norm = df[df.label == 0].index[:4].tolist()
selection = idx_frac + idx_norm

ds = FractureDataset(SPLITS / "train.csv", transforms=get_train_transforms())

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, i in zip(axes.flat, selection):
    image, label = ds[i]
    img = image.permute(1, 2, 0).numpy()
    img = (img * STD + MEAN).clip(0, 1)
    ax.imshow(img)
    ax.set_title("FRACTURE" if label == 1 else "normal")
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "sanity_check.png", dpi=100)
print(f"Figure sauvegardee : {OUT / 'sanity_check.png'}")
