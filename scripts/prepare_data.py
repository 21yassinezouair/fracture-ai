"""
Phase 2 - Construction de l'index unifie et des splits train/val/test.
Dataset : FracAtlas (public).
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- Chemins ---
RAW = Path.home() / "fracture-ai" / "data" / "raw" / "FracAtlas"
CSV = RAW / "dataset.csv"
IMAGES = RAW / "images"
OUT = Path.home() / "fracture-ai" / "data" / "processed"
SPLITS = Path.home() / "fracture-ai" / "data" / "splits"
OUT.mkdir(parents=True, exist_ok=True)
SPLITS.mkdir(parents=True, exist_ok=True)

SEED = 42

# --- 1. Lire le CSV FracAtlas ---
df = pd.read_csv(CSV)
print(f"Images dans le CSV : {len(df)}")

# --- 2. Construire le chemin de chaque image ---
def build_path(row):
    sub = "Fractured" if row["fractured"] == 1 else "Non_fractured"
    return str(IMAGES / sub / row["image_id"])

df["image_path"] = df.apply(build_path, axis=1)

# --- 3. Le label (0 = pas de fracture, 1 = fracture) ---
df["label"] = df["fractured"].astype(int)

# --- 4. La localisation : MS = membre superieur, MI = membre inferieur ---
def body_site(row):
    if row["hand"] == 1 or row["shoulder"] == 1:
        return "MS"
    if row["leg"] == 1 or row["hip"] == 1:
        return "MI"
    return "autre"

df["body_site"] = df.apply(body_site, axis=1)

# --- 5. patient_id : FracAtlas n'en fournit pas ---
# On utilise image_id comme identifiant (split par image).
# LIMITE a declarer dans la these.
df["patient_id"] = df["image_id"]
df["source"] = "fracatlas"

# --- 6. Verifier que les images existent vraiment sur le disque ---
exists = df["image_path"].apply(lambda p: Path(p).exists())
print(f"Images trouvees sur le disque : {exists.sum()} / {len(df)}")
df = df[exists].reset_index(drop=True)

# --- 7. Ecrire l'index unifie ---
index = df[["image_path", "label", "patient_id", "body_site", "source"]]
index.to_csv(OUT / "index.csv", index=False)
print(f"index.csv cree : {len(index)} lignes")

# --- 8. Split stratifie 70 / 15 / 15 ---
train, temp = train_test_split(
    index, test_size=0.30, stratify=index["label"], random_state=SEED)
val, test = train_test_split(
    temp, test_size=0.50, stratify=temp["label"], random_state=SEED)

train.to_csv(SPLITS / "train.csv", index=False)
val.to_csv(SPLITS / "val.csv", index=False)
test.to_csv(SPLITS / "test.csv", index=False)

# --- 9. Verifications ---
print("\n--- Repartition ---")
for name, d in [("train", train), ("val", val), ("test", test)]:
    print(f"{name}: {len(d)} images, {d['label'].mean()*100:.1f}% fractures")

# Anti-fuite : aucune image partagee entre les splits
assert set(train["image_path"]) & set(test["image_path"]) == set()
assert set(train["image_path"]) & set(val["image_path"]) == set()
assert set(val["image_path"]) & set(test["image_path"]) == set()
print("\nOK - Aucune fuite entre les splits.")
