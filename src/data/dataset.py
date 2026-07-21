"""
FractureDataset : lit le CSV d'un split, charge chaque image et son label.
"""
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

class FractureDataset(Dataset):
    def __init__(self, csv_path, transforms=None):
        self.df = pd.read_csv(csv_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Lire l'image (OpenCV lit en BGR) et convertir en RGB
        image = cv2.imread(row["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Appliquer les transformations
        if self.transforms:
            image = self.transforms(image=image)["image"]
        label = torch.tensor(row["label"], dtype=torch.float32)
        return image, label
