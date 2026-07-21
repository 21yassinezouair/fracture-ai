"""
Transformations d'images.
- Train : redimensionnement + augmentation (variete pour eviter l'overfitting).
- Val/Test : redimensionnement seulement (aucune augmentation).
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Normalisation ImageNet (obligatoire car les modeles sont pre-entraines dessus)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SIZE = 224

def get_train_transforms():
    return A.Compose([
        A.Resize(SIZE, SIZE),
        A.HorizontalFlip(p=0.5),          # un os gauche/droit reste un os
        A.Rotate(limit=15, p=0.5),        # variabilite de positionnement
        A.RandomBrightnessContrast(p=0.3),# variabilite des equipements RX
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(SIZE, SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
