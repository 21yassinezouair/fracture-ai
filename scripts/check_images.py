"""
Verifie que toutes les images du dataset sont lisibles.
Liste les images corrompues (a exclure).
"""
import cv2
import pandas as pd
from pathlib import Path

INDEX = Path.home() / "fracture-ai" / "data" / "processed" / "index.csv"
df = pd.read_csv(INDEX)

corrompues = []
for path in df["image_path"]:
    img = cv2.imread(path)
    if img is None or img.size == 0:
        corrompues.append(path)

print(f"Images totales   : {len(df)}")
print(f"Images corrompues : {len(corrompues)}")
for p in corrompues:
    print("  ->", p)
